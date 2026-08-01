from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import NoReturn, Protocol, TypeVar

import sherpa_onnx

from sttx import __version__
from sttx.asr import (
    FloatSamples,
    RecognitionResult,
    RecognitionStream,
    Recognizer,
    TranscriptionError,
    VadSegment,
    VoiceActivityDetector,
    transcribe,
)
from sttx.audio import AudioEnvironmentError, PreparedAudio, normalize_media
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
RecognizerT = TypeVar("RecognizerT")
VadT = TypeVar("VadT")
RecognizerT_co = TypeVar("RecognizerT_co", covariant=True)
VadT_co = TypeVar("VadT_co", covariant=True)
RecognizerT_contra = TypeVar("RecognizerT_contra", contravariant=True)
VadT_contra = TypeVar("VadT_contra", contravariant=True)


class RecognizerFactory(Protocol[RecognizerT_co]):
    def __call__(self, bundle: ModelBundle, /) -> RecognizerT_co: ...


class VadFactory(Protocol[VadT_co]):
    def __call__(self, bundle: ModelBundle, /) -> VadT_co: ...


class TranscribeFn(Protocol[RecognizerT_contra, VadT_contra]):
    def __call__(
        self,
        audio: PreparedAudio,
        /,
        *,
        recognizer: RecognizerT_contra,
        vad: VadT_contra,
    ) -> Transcript: ...


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


@dataclass(frozen=True, slots=True)
class _VerboseProgress:
    enabled: bool
    started: float

    def report(self, message: str) -> None:
        if self.enabled:
            elapsed = time.perf_counter() - self.started
            print(f"[+{elapsed:.3f}s] {message}", file=sys.stderr)


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
    parser = _Parser(prog="sttx", description="Transcribe one media file locally.")
    parser.add_argument("media", type=Path)
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument("-d", "--outdir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _default_make_recognizer(
    bundle: ModelBundle,
) -> Recognizer[RecognitionStream, RecognitionResult]:
    return _make_recognizer_from_bundle(bundle)


def _default_make_vad(bundle: ModelBundle) -> VoiceActivityDetector:
    return _make_vad_from_bundle(bundle)


def run(
    argv: Sequence[str] | None = None,
    *,
    _normalize_media: Callable[[Path], PreparedAudio] = normalize_media,
    _resolve_bundle: Callable[[Path | None], ModelBundle] = resolve_bundle_cancellable,
    _make_recognizer: RecognizerFactory[RecognizerT] = _default_make_recognizer,
    _make_vad: VadFactory[VadT] = _default_make_vad,
    _transcribe: TranscribeFn[RecognizerT, VadT] = transcribe,
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
        return _run(
            argv,
            _normalize_media=_normalize_media,
            _resolve_bundle=_resolve_bundle,
            _make_recognizer=_make_recognizer,
            _make_vad=_make_vad,
            _transcribe=_transcribe,
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
    _normalize_media: Callable[[Path], PreparedAudio] = normalize_media,
    _resolve_bundle: Callable[[Path | None], ModelBundle] = resolve_bundle_cancellable,
    _make_recognizer: RecognizerFactory[RecognizerT] = _default_make_recognizer,
    _make_vad: VadFactory[VadT] = _default_make_vad,
    _transcribe: TranscribeFn[RecognizerT, VadT] = transcribe,
    _output_paths: Callable[[Path, Path | None, str | None, Path], OutputPaths] = output_paths,
    _write_outputs: Callable[[Transcript, OutputPaths], None] = write_outputs,
    _cwd: Path | None = None,
) -> int:
    parser = build_parser()
    try:
        namespace = parser.parse_args(argv)
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
    verbose = namespace.verbose

    try:
        _validate_input(media)
        paths = _output_paths(media, outdir, output, cwd)
    except (AudioEnvironmentError, OutputPathError) as error:
        _print_error(error)
        return ENVIRONMENT_ERROR
    except (argparse.ArgumentTypeError, CliRuntimeError) as error:
        _print_error(error)
        return USAGE_OR_RUNTIME_ERROR

    started = time.perf_counter()
    progress = _VerboseProgress(enabled=verbose, started=started)
    progress.report(f"input path={media}")
    progress.report(f"output json={paths.json_path} txt={paths.txt_path}")
    transcript: Transcript | None = None
    try:
        progress.report("normalize start")
        with _normalize_media(media) as prepared:
            progress.report(f"normalize complete duration={prepared.duration:.2f}s")
            progress.report("models resolve start")
            bundle = _resolve_bundle(model_dir)
            progress.report("models resolve complete")
            progress.report("recognizer initialize start")
            recognizer = _make_recognizer(bundle)
            progress.report("recognizer initialize complete")
            progress.report("VAD initialize start")
            vad = _make_vad(bundle)
            progress.report("VAD initialize complete")
            progress.report("transcribe start")
            transcript = _transcribe_with_error_boundary(
                prepared,
                recognizer,
                vad,
                _transcribe,
            )
            progress.report(f"transcribe complete language={transcript.language} segments={len(transcript.segments)}")
    except (
        AudioEnvironmentError,
        ModelEnvironmentError,
        OutputPathError,
        OutputWriteError,
    ) as error:
        _print_error(error)
        return ENVIRONMENT_ERROR
    except (argparse.ArgumentTypeError, CliRuntimeError, TranscriptionError) as error:
        _print_error(error)
        return USAGE_OR_RUNTIME_ERROR
    if transcript is None:
        _print_error(CliRuntimeError("transcription decode failed: no transcript produced"))
        return USAGE_OR_RUNTIME_ERROR

    try:
        progress.report("write outputs start")
        _write_outputs(transcript, paths)
        progress.report("write outputs complete")
    except OutputWriteError as error:
        _print_error(error)
        return ENVIRONMENT_ERROR
    except CliRuntimeError as error:
        _print_error(error)
        return USAGE_OR_RUNTIME_ERROR

    if not transcript.segments:
        print("warning: no speech detected", file=sys.stderr)
    if verbose:
        _print_progress(transcript, started)
    print(paths.json_path)
    print(paths.txt_path)
    return SUCCESS


def main() -> None:
    raise SystemExit(run())


def _transcribe_with_error_boundary(
    prepared: PreparedAudio,
    recognizer: RecognizerT,
    vad: VadT,
    transcribe_fn: TranscribeFn[RecognizerT, VadT],
) -> Transcript:
    try:
        return transcribe_fn(
            prepared,
            recognizer=recognizer,
            vad=vad,
        )
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


def _print_progress(transcript: Transcript, started: float) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rtf = elapsed / max(transcript.duration, 1e-9)
    print(
        f"complete duration={transcript.duration:.2f}s rtf={rtf:.3f}",
        file=sys.stderr,
    )
