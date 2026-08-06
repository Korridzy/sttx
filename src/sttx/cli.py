from __future__ import annotations

from _thread import LockType
import argparse
import json
import os
import signal
import shlex
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from types import FrameType, TracebackType
from typing import Generic, NoReturn, Protocol, Self, TypeVar, assert_never

import sherpa_onnx

from sttx import __version__
from sttx.asr import (
    FloatSamples,
    ProgressCallback,
    RecognitionResult,
    RecognitionStream,
    Recognizer,
    TranscriptionError,
    VadSegment,
    VoiceActivityDetector,
    transcribe,
)
from sttx.audio import AudioEnvironmentError, PreparedAudio, ffmpeg_argv, normalize_media
from sttx.asr_events import (
    ActivityCallback,
    AsrActivity,
    DecodeFinished,
    DecodeStarted,
    LanguageReported,
    ScanAdvanced,
    ScanFinished,
    ScanStarted,
    TranscriptionSummary,
    VadSegmentReady,
    WordCountUpdated,
)
from sttx.model import (
    ModelBundle,
    ModelEnvironmentError,
    resolve_bundle_cancellable,
)
from sttx.output import (
    OutputPathError,
    OutputPaths,
    OutputWriteError,
    Transcript,
    output_paths,
    write_outputs,
)

SUCCESS: int = 0
ENVIRONMENT_ERROR: int = 1
USAGE_OR_RUNTIME_ERROR: int = 2
SAMPLE_RATE: int = 16_000
PROGRESS_BAR_WIDTH: int = 20
LIVE_PROGRESS_TICK_SECONDS: float = 0.1
LIVE_PROGRESS_RENDER_PREFIX: bytes = b"\x00sttx-live-progress-render\x00"
LIVE_PROGRESS_RENDER_SUFFIX: bytes = b"\x00sttx-live-progress-end\x00"
LIVE_PROGRESS_FINISH: bytes = b"\x00sttx-live-progress-finish\x00"
LIVE_PROGRESS_STOP_FACTORY: Callable[[], threading.Event] = threading.Event
JsonLogField = str | int | float | bool | None | list[str]
PreparedAudioT = TypeVar("PreparedAudioT", bound="_PreparedAudioResource")
RecognizerT = TypeVar("RecognizerT")
VadT = TypeVar("VadT")
PreparedAudioT_contra = TypeVar(
    "PreparedAudioT_contra",
    bound="_PreparedAudioResource",
    contravariant=True,
)
RecognizerT_contra = TypeVar("RecognizerT_contra", contravariant=True)
VadT_contra = TypeVar("VadT_contra", contravariant=True)


def _format_duration(duration_seconds: float) -> str:
    total_milliseconds = round(duration_seconds * 1_000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class _PreparedAudioResource(Protocol):
    path: Path
    sample_count: int

    @property
    def duration(self) -> float: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...


class _Transcriber(
    Protocol[PreparedAudioT_contra, RecognizerT_contra, VadT_contra],
):
    def __call__(
        self,
        audio: PreparedAudioT_contra,
        /,
        *,
        recognizer: RecognizerT_contra,
        vad: VadT_contra,
        progress: ProgressCallback | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript: ...


@dataclass(frozen=True, slots=True)
class RunnerDependencies(Generic[PreparedAudioT, RecognizerT, VadT]):
    normalize_media: Callable[[Path], PreparedAudioT]
    resolve_bundle: Callable[[Path | None], ModelBundle]
    make_recognizer: Callable[[ModelBundle], RecognizerT]
    make_vad: Callable[[ModelBundle], VadT]
    transcribe: _Transcriber[PreparedAudioT, RecognizerT, VadT]


@dataclass(frozen=True, slots=True)
class CliRuntimeError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ShutdownRequested(BaseException):
    signum: int

    def __str__(self) -> str:
        return f"cancelled by {signal.Signals(self.signum).name}"


class _NativeStderrRelay:
    __slots__: tuple[str, ...] = (
        "destination_fd",
        "pipe_read_fd",
        "pipe_write_fd",
        "thread",
        "line_open",
        "live_progress_visible",
        "pending",
    )
    destination_fd: int
    pipe_read_fd: int
    pipe_write_fd: int
    thread: Thread
    line_open: bool
    live_progress_visible: bool
    pending: bytes

    def __init__(self) -> None:
        self.destination_fd = os.dup(2)
        self.pipe_read_fd, self.pipe_write_fd = os.pipe()
        self.thread = Thread(target=self._drain_native_stderr, daemon=False)
        self.line_open = False
        self.live_progress_visible = False
        self.pending = b""

    def start(self) -> None:
        self.thread.start()
        sys.stderr.flush()
        os.dup2(self.pipe_write_fd, 2)
        os.close(self.pipe_write_fd)

    def close(self) -> None:
        sys.stderr.flush()
        os.dup2(self.destination_fd, 2)
        self.thread.join()
        os.close(self.pipe_read_fd)
        os.close(self.destination_fd)

    def render_live_progress(self, message: str) -> None:
        os.write(
            2,
            LIVE_PROGRESS_RENDER_PREFIX
            + message.encode()
            + LIVE_PROGRESS_RENDER_SUFFIX,
        )

    def finish_live_progress(self) -> None:
        os.write(2, LIVE_PROGRESS_FINISH)

    def _drain_native_stderr(self) -> None:
        while payload := os.read(self.pipe_read_fd, 4_096):
            self.pending += payload
            self._drain_pending()
        self._forward_native(self.pending)

    def _drain_pending(self) -> None:
        while self.pending:
            marker_index = self._next_marker_index()
            if marker_index is None:
                keep = self._incomplete_marker_tail_length()
                native_length = len(self.pending) - keep
                if native_length <= 0:
                    return
                self._forward_native(self.pending[:native_length])
                self.pending = self.pending[native_length:]
                continue
            if marker_index > 0:
                self._forward_native(self.pending[:marker_index])
                self.pending = self.pending[marker_index:]
                continue
            if self.pending.startswith(LIVE_PROGRESS_FINISH):
                self._finish_live_progress()
                self.pending = self.pending[len(LIVE_PROGRESS_FINISH) :]
                continue
            suffix_index = self.pending.find(
                LIVE_PROGRESS_RENDER_SUFFIX,
                len(LIVE_PROGRESS_RENDER_PREFIX),
            )
            if suffix_index < 0:
                return
            message_start = len(LIVE_PROGRESS_RENDER_PREFIX)
            message = self.pending[message_start:suffix_index]
            self._render_live_progress(message)
            self.pending = self.pending[
                suffix_index + len(LIVE_PROGRESS_RENDER_SUFFIX) :
            ]

    def _next_marker_index(self) -> int | None:
        render_index = self.pending.find(LIVE_PROGRESS_RENDER_PREFIX)
        finish_index = self.pending.find(LIVE_PROGRESS_FINISH)
        if render_index < 0:
            return finish_index if finish_index >= 0 else None
        if finish_index < 0:
            return render_index
        return min(render_index, finish_index)

    def _incomplete_marker_tail_length(self) -> int:
        longest_tail = 0
        for marker in (LIVE_PROGRESS_RENDER_PREFIX, LIVE_PROGRESS_FINISH):
            maximum_length = min(len(self.pending), len(marker) - 1)
            for length in range(maximum_length, 0, -1):
                if self.pending.endswith(marker[:length]):
                    longest_tail = max(longest_tail, length)
                    break
        return longest_tail

    def _render_live_progress(self, message: bytes) -> None:
        if self.line_open and not self.live_progress_visible:
            self._write(b"\n")
        self._write(message)
        self.line_open = True
        self.live_progress_visible = True

    def _finish_live_progress(self) -> None:
        if self.live_progress_visible:
            self._write(b"\n")
            self.line_open = False
            self.live_progress_visible = False

    def _forward_native(self, payload: bytes) -> None:
        if not payload:
            return
        self._finish_live_progress()
        self._write(payload)
        self.line_open = not payload.endswith(b"\n")

    def _write(self, payload: bytes) -> None:
        remaining = payload
        while remaining:
            written = os.write(self.destination_fd, remaining)
            remaining = remaining[written:]


class _VerboseProgress:
    __slots__: tuple[str, ...] = (
        "verbosity",
        "debug",
        "json_logging",
        "started",
        "last_percent",
        "last_progress_at",
        "transcription_started",
        "decode_started",
        "decode_elapsed",
        "live_progress_width",
        "live_progress_lock",
        "live_progress_stop",
        "live_progress_thread",
        "live_progress_body",
        "native_stderr_relay",
    )
    verbosity: int
    debug: bool
    json_logging: bool
    started: float
    last_percent: int
    last_progress_at: float | None
    transcription_started: float
    decode_started: float
    decode_elapsed: float
    live_progress_width: int
    live_progress_lock: LockType
    live_progress_stop: threading.Event | None
    live_progress_thread: threading.Thread | None
    live_progress_body: str | None
    native_stderr_relay: _NativeStderrRelay | None

    def __init__(
        self,
        verbosity: int,
        debug: bool,
        json_logging: bool,
        started: float,
    ) -> None:
        self.verbosity = verbosity
        self.debug = debug
        self.json_logging = json_logging
        self.started = started
        self.last_percent = 0
        self.last_progress_at = None
        self.transcription_started = started
        self.decode_started = started
        self.decode_elapsed = 0.0
        self.live_progress_width = 0
        self.live_progress_lock = threading.Lock()
        self.live_progress_stop = None
        self.live_progress_thread = None
        self.live_progress_body = None
        self.native_stderr_relay = None

    def report(self, message: str, /, *, event: str, **fields: JsonLogField) -> None:
        if self.verbosity > 0:
            self._emit(message, event=event, **fields)

    def report_debug(self, message: str, /, *, event: str, **fields: JsonLogField) -> None:
        if self.debug:
            self._emit(f"debug {message}", event=event, **fields)

    def report_stage_duration(self, stage: str, stage_started: float) -> None:
        duration = time.perf_counter() - stage_started
        self.report_debug(
            f"stage={stage} duration={_format_duration(duration)}",
            event="stage_duration",
            stage=stage,
            duration_seconds=duration,
        )

    def report_error(self, stage: str, error: BaseException) -> None:
        self._finish_live_progress()
        if self.json_logging:
            self._emit(
                str(error),
                event="error",
                stage=stage,
                exception_type=type(error).__name__,
                message=str(error),
            )
            return
        print(f"error: {error}", file=sys.stderr)

    def report_warning(self, message: str) -> None:
        self._finish_live_progress()
        if self.json_logging:
            self._emit(message, event="warning")
            return
        print(f"warning: {message}", file=sys.stderr)

    def _emit(self, message: str, /, *, event: str, **fields: JsonLogField) -> None:
        self._finish_live_progress()
        elapsed = time.perf_counter() - self.started
        if self.json_logging:
            print(
                json.dumps(
                    {"event": event, "elapsed_seconds": elapsed, **fields},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return
        print(f"[+{_format_duration(elapsed)}] {message}", file=sys.stderr)

    def _finish_live_progress(self) -> None:
        with self.live_progress_lock:
            stop = self.live_progress_stop
            thread = self.live_progress_thread
            self.live_progress_stop = None
            self.live_progress_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.ident is not None:
            thread.join()
        with self.live_progress_lock:
            if self.live_progress_width > 0:
                if self.native_stderr_relay is None:
                    print(file=sys.stderr)
                else:
                    self.native_stderr_relay.finish_live_progress()
                self.live_progress_width = 0
            self.live_progress_body = None
            native_stderr_relay = self.native_stderr_relay
            self.native_stderr_relay = None
        if native_stderr_relay is not None:
            native_stderr_relay.close()

    def start_transcription(self) -> None:
        self._finish_live_progress()
        self.transcription_started = time.perf_counter()
        self.last_percent = 0
        self.last_progress_at = None
        self.decode_started = self.transcription_started
        self.decode_elapsed = 0.0
        if self.verbosity <= 0 or self.debug or self.json_logging or not sys.stderr.isatty():
            return
        stop = LIVE_PROGRESS_STOP_FACTORY()
        thread = threading.Thread(
            target=lambda: self._spin_live_progress(stop),
            daemon=False,
        )
        with self.live_progress_lock:
            native_stderr_relay = _NativeStderrRelay()
            native_stderr_relay.start()
            self.native_stderr_relay = native_stderr_relay
            self.live_progress_stop = stop
            self.live_progress_thread = thread
        thread.start()

    def stop_transcription(self) -> None:
        self._finish_live_progress()

    def _spin_live_progress(self, stop: threading.Event) -> None:
        while not stop.wait(LIVE_PROGRESS_TICK_SECONDS):
            with self.live_progress_lock:
                if self.live_progress_stop is stop:
                    self._tick_live_progress_locked()

    def _tick_live_progress_locked(self) -> None:
        if self.live_progress_body is None:
            return
        self._render_live_progress_locked()

    def _update_live_progress(self, body: str) -> None:
        with self.live_progress_lock:
            self.live_progress_body = body
            self._render_live_progress_locked()

    def _render_live_progress_locked(self) -> None:
        if self.live_progress_body is None:
            return
        elapsed = time.perf_counter() - self.started
        message = f"[+{_format_duration(elapsed)}] {self.live_progress_body}"
        padding = " " * max(self.live_progress_width - len(message), 0)
        rendered = f"\r{message}{padding}\r"
        if self.native_stderr_relay is None:
            sys.stderr.write(rendered)
            sys.stderr.flush()
        else:
            self.native_stderr_relay.render_live_progress(rendered)
        self.live_progress_width = len(message)

    def report_transcription(self, processed_samples: int, total_samples: int) -> None:
        self._report_scan_progress(
            processed_samples,
            total_samples,
            prefix="transcribe",
            event="transcribe_progress",
        )

    def report_activity(self, event: AsrActivity) -> None:
        match event:
            case ScanStarted(total_samples=total_samples):
                self.start_transcription()
                self.report_debug(
                    f"asr scan start total={total_samples} samples",
                    event="asr_scan_started",
                    total_samples=total_samples,
                )
            case ScanAdvanced(
                processed_samples=processed_samples,
                total_samples=total_samples,
            ):
                self._report_scan_progress(
                    processed_samples,
                    total_samples,
                    prefix="debug asr scan",
                    event="asr_scan_progress",
                )
            case VadSegmentReady(index=index, start_sample=start_sample, sample_count=sample_count):
                self.report_debug(
                    f"asr vad segment={index} audio={_format_duration(start_sample / SAMPLE_RATE)}+{_format_duration(sample_count / SAMPLE_RATE)}",
                    event="asr_vad_segment_ready",
                    index=index,
                    start_seconds=start_sample / SAMPLE_RATE,
                    audio_seconds=sample_count / SAMPLE_RATE,
                )
            case DecodeStarted(index=index, start_sample=start_sample, sample_count=sample_count):
                self.decode_started = time.perf_counter()
                self.report_debug(
                    f"asr decode start chunk={index} audio={_format_duration(start_sample / SAMPLE_RATE)}+{_format_duration(sample_count / SAMPLE_RATE)}",
                    event="asr_decode_started",
                    index=index,
                    start_seconds=start_sample / SAMPLE_RATE,
                    audio_seconds=sample_count / SAMPLE_RATE,
                )
            case DecodeFinished(index=index, sample_count=sample_count):
                elapsed = time.perf_counter() - self.decode_started
                self.decode_elapsed += elapsed
                audio_seconds = max(sample_count / SAMPLE_RATE, 1e-9)
                self.report_debug(
                    f"asr decode done chunk={index} elapsed={_format_duration(elapsed)} rtf={elapsed / audio_seconds:.3f}",
                    event="asr_decode_finished",
                    index=index,
                    audio_seconds=audio_seconds,
                    decode_elapsed_seconds=elapsed,
                    rtf=elapsed / audio_seconds,
                )
            case LanguageReported(language=language):
                self.report_debug(
                    f"asr language reported={language}",
                    event="asr_language_reported",
                    language=language,
                )
            case WordCountUpdated(word_count=word_count):
                self.report_debug(
                    f"asr words={word_count}",
                    event="asr_word_count_updated",
                    word_count=word_count,
                )
            case ScanFinished(total_samples=total_samples):
                self.report_debug(
                    f"asr scan complete total={total_samples} samples",
                    event="asr_scan_finished",
                    total_samples=total_samples,
                )
            case TranscriptionSummary(
                voiced_samples=voiced_samples,
                vad_segments=vad_segments,
                decoded_samples=decoded_samples,
                decode_chunks=decode_chunks,
                word_count=word_count,
                language=language,
                transcript_segments=transcript_segments,
            ):
                decoded_seconds = max(decoded_samples / SAMPLE_RATE, 1e-9)
                self.report_debug(
                    " ".join(
                        (
                            "asr summary",
                            f"vad_segments={vad_segments}",
                            f"decode_chunks={decode_chunks}",
                            f"words={word_count}",
                            f"transcript_segments={transcript_segments}",
                            f"language={language}",
                        )
                    ),
                    event="asr_summary",
                    vad_segments=vad_segments,
                    decode_chunks=decode_chunks,
                    word_count=word_count,
                    transcript_segments=transcript_segments,
                    language=language,
                    voiced_seconds=voiced_samples / SAMPLE_RATE,
                    decoded_seconds=decoded_samples / SAMPLE_RATE,
                )
                self.report_debug(
                    " ".join(
                        (
                            "asr timing",
                            f"voiced={_format_duration(voiced_samples / SAMPLE_RATE)}",
                            f"decoded={_format_duration(decoded_samples / SAMPLE_RATE)}",
                            f"decode_elapsed={_format_duration(self.decode_elapsed)}",
                            f"decode_rtf={self.decode_elapsed / decoded_seconds:.3f}",
                        )
                    ),
                    event="asr_timing",
                    voiced_seconds=voiced_samples / SAMPLE_RATE,
                    decoded_seconds=decoded_samples / SAMPLE_RATE,
                    decode_elapsed_seconds=self.decode_elapsed,
                    decode_rtf=self.decode_elapsed / decoded_seconds,
                )
            case unreachable:
                assert_never(unreachable)

    def _report_scan_progress(
        self,
        processed_samples: int,
        total_samples: int,
        *,
        prefix: str,
        event: str,
    ) -> None:
        percent = processed_samples * 100 // total_samples
        if percent <= self.last_percent:
            return
        now = time.perf_counter()
        if (
            percent < 100
            and self.last_progress_at is not None
            and now - self.last_progress_at < 1.0
        ):
            return
        self.last_percent = percent
        self.last_progress_at = now
        elapsed = max(now - self.transcription_started, 1e-9)
        processed_seconds = processed_samples / SAMPLE_RATE
        total_seconds = total_samples / SAMPLE_RATE
        rtf = elapsed / processed_seconds
        eta = (total_seconds - processed_seconds) * rtf
        progress_details = " ".join(
            (
                f"audio={_format_duration(processed_seconds)}/{_format_duration(total_seconds)}",
                f"rtf={rtf:.3f}",
                f"eta={_format_duration(eta)}",
            )
        )
        if not self.debug and not self.json_logging:
            completed_units = percent * PROGRESS_BAR_WIDTH // 100
            progress_bar = "#" * completed_units + "-" * (PROGRESS_BAR_WIDTH - completed_units)
            if self.native_stderr_relay is not None:
                self._update_live_progress(
                    " ".join(
                        (
                            f"[{progress_bar}]",
                            f"{percent}%",
                            f"{_format_duration(processed_seconds)}/{_format_duration(total_seconds)}",
                            f"rtf={rtf:.3f}",
                            f"eta={_format_duration(eta)}",
                        )
                    )
                )
                return
            message = f"{prefix} [{progress_bar}] progress={percent}%\n  {progress_details}"
        else:
            message = f"{prefix} progress={percent}% {progress_details}"
        self.report(
            message,
            event=event,
            percent=percent,
            audio_seconds=processed_seconds,
            audio_total_seconds=total_seconds,
            transcription_elapsed_seconds=elapsed,
            rtf=rtf,
            eta_seconds=eta,
        )


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliRuntimeError(f"{self.prog}: error: {message}")


@dataclass(frozen=True, slots=True)
class _RecognizerAdapter:
    raw: sherpa_onnx.OfflineRecognizer

    def create_stream(self) -> RecognitionStream:
        return self.raw.create_stream()

    def decode_stream(self, stream: RecognitionStream) -> None:
        self.raw.decode_stream(stream)


@dataclass(frozen=True, slots=True)
class _VoiceActivityDetectorAdapter:
    raw: sherpa_onnx.VoiceActivityDetector

    def accept_waveform(self, samples: FloatSamples) -> None:
        self.raw.accept_waveform(samples)

    def flush(self) -> None:
        self.raw.flush()

    def empty(self) -> bool:
        return self.raw.empty()

    @property
    def front(self) -> VadSegment:
        return self.raw.front

    def pop(self) -> None:
        self.raw.pop()


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="sttx",
        description="Transcribe one media file locally.",
        allow_abbrev=False,
    )
    parser.add_argument("media", type=Path)
    parser.add_argument(
        "-o",
        "--output-name",
        dest="output",
        metavar="STEM",
        help="filename stem for JSON/TXT outputs, not a path",
    )
    parser.add_argument(
        "-d",
        "--outdir",
        type=Path,
        metavar="DIR",
        help="directory for JSON/TXT outputs; relative paths use the current directory (default: ./transcriptions)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        metavar="DIR",
        help="use an offline local model bundle; it must contain all required assets",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="write stage and live transcription progress to stderr",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="write internal diagnostics and unexpected-error tracebacks to stderr",
    )
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default="text",
        help="format diagnostic output on stderr",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def run(
    argv: Sequence[str] | None = None,
    *,
    _dependencies: RunnerDependencies[PreparedAudioT, RecognizerT, VadT] | None = None,
    _output_paths: Callable[[Path, Path | None, str | None, Path], OutputPaths] = output_paths,
    _write_outputs: Callable[[Transcript, OutputPaths], None] = write_outputs,
    _cwd: Path | None = None,
) -> int:
    previous_handlers: dict[
        int,
        int | Callable[[int, FrameType | None], None] | None,
    ] = {}
    install_handlers = threading.current_thread() is threading.main_thread()
    shutdown_signum: int | None = None

    def request_shutdown(signum: int, frame: FrameType | None) -> None:
        nonlocal shutdown_signum
        del frame
        if shutdown_signum is None:
            shutdown_signum = signum
            raise ShutdownRequested(signum)

    try:
        if install_handlers:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, request_shutdown)
        if _dependencies is None:
            return _run_production(
                argv,
                _output_paths=_output_paths,
                _write_outputs=_write_outputs,
                _cwd=_cwd,
            )
        return _run(
            argv,
            _dependencies=_dependencies,
            _output_paths=_output_paths,
            _write_outputs=_write_outputs,
            _cwd=_cwd,
        )
    except ShutdownRequested as error:
        _print_error(error)
        return 128 + error.signum
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _run(
    argv: Sequence[str] | None = None,
    *,
    _dependencies: RunnerDependencies[PreparedAudioT, RecognizerT, VadT],
    _output_paths: Callable[[Path, Path | None, str | None, Path], OutputPaths] = output_paths,
    _write_outputs: Callable[[Transcript, OutputPaths], None] = write_outputs,
    _cwd: Path | None = None,
) -> int:
    parser = build_parser()
    try:
        namespace = parser.parse_args(argv)
        if namespace.verbose > 1:
            raise CliRuntimeError("--verbose may be specified once")
    except CliRuntimeError as error:
        parser.print_usage(sys.stderr)
        _print_error(error)
        return USAGE_OR_RUNTIME_ERROR
    except SystemExit as error:
        return _system_exit_code(error)

    cwd = Path.cwd() if _cwd is None else _cwd
    media = namespace.media
    output = namespace.output
    outdir = namespace.outdir
    model_dir = namespace.model_dir
    debug = namespace.debug
    verbosity = max(namespace.verbose, int(debug))
    json_logging = namespace.log_format == "json"
    started = time.perf_counter()
    progress = _VerboseProgress(
        verbosity=verbosity,
        debug=debug,
        json_logging=json_logging,
        started=started,
    )
    stage = "validate"
    progress.report_debug(
        f"configuration media={media} outdir={outdir} model_dir={model_dir} verbosity={namespace.verbose}",
        event="configuration",
        media_path=str(media),
        outdir_path=str(outdir) if outdir is not None else None,
        model_dir_path=str(model_dir) if model_dir is not None else None,
        verbosity=namespace.verbose,
        log_format=namespace.log_format,
    )

    try:
        _validate_input(media)
        stage = "output-paths"
        paths = _output_paths(media, outdir, output, cwd)
    except (AudioEnvironmentError, OutputPathError) as error:
        _report_failure(progress, stage, error)
        progress.report_error(stage, error)
        return ENVIRONMENT_ERROR
    except (argparse.ArgumentTypeError, CliRuntimeError) as error:
        _report_failure(progress, stage, error)
        progress.report_error(stage, error)
        return USAGE_OR_RUNTIME_ERROR
    except Exception as error:  # noqa: BROAD_EXCEPT_OK
        if debug:
            _print_debug_traceback(progress, stage, error)
            return USAGE_OR_RUNTIME_ERROR
        raise

    progress.report(f"input path={media}", event="input", media_path=str(media))
    progress.report(
        f"output json={paths.json_path} txt={paths.txt_path}",
        event="output_paths",
        json_path=str(paths.json_path),
        txt_path=str(paths.txt_path),
    )
    transcript: Transcript | None = None
    try:
        stage = "normalize"
        stage_started = time.perf_counter()
        progress.report("normalize start", event="stage_started", stage=stage)
        with _dependencies.normalize_media(media) as prepared:
            progress.report(
                f"normalize complete duration={_format_duration(prepared.duration)}",
                event="audio_ready",
                audio_seconds=prepared.duration,
                sample_count=prepared.sample_count,
                sample_rate=SAMPLE_RATE,
            )
            progress.report_debug(
                f"ffmpeg argv={shlex.join(ffmpeg_argv(media, prepared.path))}",
                event="ffmpeg_invocation",
                argv=list(ffmpeg_argv(media, prepared.path)),
            )
            progress.report_debug(
                f"normalized_wav={prepared.path}",
                event="normalized_audio_path",
                wav_path=str(prepared.path),
            )
            progress.report_stage_duration(stage, stage_started)
            stage = "models"
            stage_started = time.perf_counter()
            progress.report("models resolve start", event="stage_started", stage=stage)
            bundle = _dependencies.resolve_bundle(model_dir)
            progress.report("models resolve complete", event="stage_completed", stage=stage)
            _report_bundle_debug(
                progress,
                bundle,
                source="explicit" if model_dir is not None else bundle.source,
            )
            progress.report_stage_duration(stage, stage_started)
            stage = "recognizer"
            stage_started = time.perf_counter()
            progress.report("recognizer initialize start", event="stage_started", stage=stage)
            recognizer = _dependencies.make_recognizer(bundle)
            progress.report("recognizer initialize complete", event="stage_completed", stage=stage)
            progress.report_stage_duration(stage, stage_started)
            stage = "vad"
            stage_started = time.perf_counter()
            progress.report("VAD initialize start", event="stage_started", stage=stage)
            vad = _dependencies.make_vad(bundle)
            progress.report("VAD initialize complete", event="stage_completed", stage=stage)
            progress.report_stage_duration(stage, stage_started)
            stage = "transcribe"
            stage_started = time.perf_counter()
            progress.report("transcribe start", event="stage_started", stage=stage)
            transcription_progress = (
                progress.report_transcription if verbosity > 0 and not debug else None
            )
            transcription_activity = progress.report_activity if debug else None
            try:
                progress.start_transcription()
                transcript = _transcribe_with_error_boundary(
                    lambda: _dependencies.transcribe(
                        prepared,
                        recognizer=recognizer,
                        vad=vad,
                        progress=transcription_progress,
                        activity=transcription_activity,
                    )
                )
            finally:
                progress.stop_transcription()
            progress.report(
                f"transcribe complete language={transcript.language} segments={len(transcript.segments)}",
                event="transcription_completed",
                language=transcript.language,
                segment_count=len(transcript.segments),
                word_count=len(transcript.text.split()),
            )
            progress.report_stage_duration(stage, stage_started)
    except ShutdownRequested:
        progress.stop_transcription()
        raise
    except (
        AudioEnvironmentError,
        ModelEnvironmentError,
        OutputPathError,
        OutputWriteError,
    ) as error:
        _report_failure(progress, stage, error)
        progress.report_error(stage, error)
        return ENVIRONMENT_ERROR
    except (argparse.ArgumentTypeError, CliRuntimeError, TranscriptionError) as error:
        _report_failure(progress, stage, error)
        progress.report_error(stage, error)
        return USAGE_OR_RUNTIME_ERROR
    except Exception as error:  # noqa: BROAD_EXCEPT_OK
        if debug:
            _print_debug_traceback(progress, stage, error)
            return USAGE_OR_RUNTIME_ERROR
        raise
    if transcript is None:
        _report_failure(
            progress,
            stage,
            CliRuntimeError("transcription decode failed: no transcript produced"),
        )
        progress.report_error(
            stage,
            CliRuntimeError("transcription decode failed: no transcript produced"),
        )
        return USAGE_OR_RUNTIME_ERROR

    try:
        stage = "write"
        stage_started = time.perf_counter()
        progress.report("write outputs start", event="stage_started", stage=stage)
        _write_outputs(transcript, paths)
        progress.report("write outputs complete", event="stage_completed", stage=stage)
        progress.report_stage_duration(stage, stage_started)
    except OutputWriteError as error:
        _report_failure(progress, stage, error)
        progress.report_error(stage, error)
        return ENVIRONMENT_ERROR
    except CliRuntimeError as error:
        _report_failure(progress, stage, error)
        progress.report_error(stage, error)
        return USAGE_OR_RUNTIME_ERROR
    except Exception as error:  # noqa: BROAD_EXCEPT_OK
        if debug:
            _print_debug_traceback(progress, stage, error)
            return USAGE_OR_RUNTIME_ERROR
        raise

    if not transcript.segments:
        progress.report_warning("no speech detected")
    if verbosity > 0:
        _print_progress(progress, transcript)
    print(paths.json_path)
    print(paths.txt_path)
    return SUCCESS


def main() -> None:
    raise SystemExit(run())


def _transcribe_with_error_boundary(
    transcribe_call: Callable[[], Transcript],
) -> Transcript:
    try:
        return transcribe_call()
    except (OSError, RuntimeError, ValueError) as error:
        raise CliRuntimeError(f"transcription decode failed: {error}") from error


def _validate_input(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.R_OK):
        raise AudioEnvironmentError(path=path, reason="input is not a readable file")


def _make_recognizer_from_bundle(
    bundle: ModelBundle,
) -> Recognizer[RecognitionStream, RecognitionResult]:
    try:
        return _RecognizerAdapter(
            sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(bundle.encoder),
                decoder=str(bundle.decoder),
                joiner=str(bundle.joiner),
                tokens=str(bundle.tokens),
                num_threads=1,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                provider="cpu",
                model_type="nemo_transducer",
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CliRuntimeError(f"recognizer construction failed: {error}") from error


def _make_vad_from_bundle(
    bundle: ModelBundle,
) -> VoiceActivityDetector:
    try:
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(bundle.silero),
                threshold=0.5,
                min_silence_duration=0.5,
                min_speech_duration=0.25,
                window_size=512,
                max_speech_duration=60,
            ),
            sample_rate=SAMPLE_RATE,
            num_threads=1,
            provider="cpu",
            debug=False,
        )
        return _VoiceActivityDetectorAdapter(
            sherpa_onnx.VoiceActivityDetector(
                config,
                buffer_size_in_seconds=60.0,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CliRuntimeError(f"VAD construction failed: {error}") from error


_PRODUCTION_DEPENDENCIES: RunnerDependencies[
    PreparedAudio,
    Recognizer[RecognitionStream, RecognitionResult],
    VoiceActivityDetector,
] = RunnerDependencies(
    normalize_media=normalize_media,
    resolve_bundle=resolve_bundle_cancellable,
    make_recognizer=_make_recognizer_from_bundle,
    make_vad=_make_vad_from_bundle,
    transcribe=transcribe,
)


def _run_production(
    argv: Sequence[str] | None,
    *,
    _output_paths: Callable[[Path, Path | None, str | None, Path], OutputPaths],
    _write_outputs: Callable[[Transcript, OutputPaths], None],
    _cwd: Path | None,
) -> int:
    return _run(
        argv,
        _dependencies=_PRODUCTION_DEPENDENCIES,
        _output_paths=_output_paths,
        _write_outputs=_write_outputs,
        _cwd=_cwd,
    )


def _system_exit_code(error: SystemExit) -> int:
    code = error.code
    if code is None:
        return SUCCESS
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return USAGE_OR_RUNTIME_ERROR


def _print_error(error: BaseException) -> None:
    print(f"error: {error}", file=sys.stderr)


def _report_bundle_debug(
    progress: _VerboseProgress,
    bundle: ModelBundle,
    *,
    source: str,
) -> None:
    progress.report_debug(
        f"model source={source}",
        event="model_source",
        source=source,
    )
    for name, path in (
        ("encoder", bundle.encoder),
        ("decoder", bundle.decoder),
        ("joiner", bundle.joiner),
        ("tokens", bundle.tokens),
        ("silero", bundle.silero),
    ):
        try:
            size = path.stat().st_size
        except OSError:
            size = "unavailable"
        progress.report_debug(
            f"model asset={name} path={path} size={size}",
            event="model_asset",
            asset_name=name,
            asset_path=str(path),
            size_bytes=size if isinstance(size, int) else None,
        )


def _report_failure(
    progress: _VerboseProgress,
    stage: str,
    error: BaseException,
) -> None:
    progress.report_debug(
        f"failure stage={stage} exception={type(error).__name__}",
        event="failure",
        stage=stage,
        exception_type=type(error).__name__,
    )


def _print_debug_traceback(
    progress: _VerboseProgress,
    stage: str,
    error: Exception,
) -> None:
    _report_failure(progress, stage, error)
    progress.report_error(stage, error)
    if progress.json_logging:
        formatted_traceback = traceback.format_exc()
        progress.report_debug(
            formatted_traceback,
            event="traceback",
            stage=stage,
            exception_type=type(error).__name__,
            traceback=formatted_traceback,
        )
        return
    traceback.print_exc(file=sys.stderr)


def _print_progress(progress: _VerboseProgress, transcript: Transcript) -> None:
    elapsed = max(time.perf_counter() - progress.started, 1e-9)
    rtf = elapsed / max(transcript.duration, 1e-9)
    progress.report(
        f"complete duration={_format_duration(transcript.duration)} rtf={rtf:.3f}",
        event="completed",
        audio_seconds=transcript.duration,
        rtf=rtf,
    )
