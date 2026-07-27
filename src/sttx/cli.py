from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import sherpa_onnx

from sttx import __version__
from sttx.asr import TranscriptionError, transcribe
from sttx.audio import AudioEnvironmentError, PreparedAudio, normalize_media
from sttx.model import ModelBundle, ModelEnvironmentError, resolve_bundle
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


class RecognizerFactory[RecognizerT](Protocol):
    def __call__(self, bundle: ModelBundle) -> RecognizerT: ...


class VadFactory[VadT](Protocol):
    def __call__(self, bundle: ModelBundle) -> VadT: ...


class TranscribeFn[RecognizerT, VadT](Protocol):
    def __call__(
        self,
        audio: PreparedAudio,
        *,
        recognizer: RecognizerT,
        vad: VadT,
    ) -> Transcript: ...


@dataclass(frozen=True, slots=True)
class CliRuntimeError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliRuntimeError(f"{self.prog}: error: {message}")


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


def run[RecognizerT, VadT](
    argv: Sequence[str] | None = None,
    *,
    _normalize_media: Callable[[Path], PreparedAudio] = normalize_media,
    _resolve_bundle: Callable[[Path | None], ModelBundle] = resolve_bundle,
    _make_recognizer: RecognizerFactory[RecognizerT] | None = None,
    _make_vad: VadFactory[VadT] | None = None,
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

    try:
        _validate_input(media)
        paths = _output_paths(media, outdir, output, cwd)
        started = time.perf_counter()
        with _normalize_media(media) as prepared:
            bundle = _resolve_bundle(model_dir)
            recognizer_factory = _make_recognizer or _make_recognizer_from_bundle
            vad_factory = _make_vad or _make_vad_from_bundle
            recognizer = recognizer_factory(bundle)
            vad = vad_factory(bundle)
            try:
                transcript = _transcribe(
                    prepared,
                    recognizer=recognizer,
                    vad=vad,
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise CliRuntimeError(f"transcription decode failed: {error}") from error
        _write_outputs(transcript, paths)
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

    if not transcript.segments:
        print("warning: no speech detected", file=sys.stderr)
    _print_progress(transcript, started)
    print(paths.json_path)
    print(paths.txt_path)
    return SUCCESS


def main() -> None:
    raise SystemExit(run())


def _validate_input(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.R_OK):
        raise AudioEnvironmentError(path=path, reason="input is not a readable file")


def _make_recognizer_from_bundle(
    bundle: ModelBundle,
) -> sherpa_onnx.OfflineRecognizer:
    try:
        return sherpa_onnx.OfflineRecognizer.from_transducer(
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
    except (OSError, RuntimeError, ValueError) as error:
        raise CliRuntimeError(f"recognizer construction failed: {error}") from error


def _make_vad_from_bundle(
    bundle: ModelBundle,
) -> sherpa_onnx.VoiceActivityDetector:
    try:
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(bundle.silero),
                threshold=0.5,
                min_silence_duration=0.5,
                min_speech_duration=0.25,
                window_size=512,
                max_speech_duration=20,
            ),
            sample_rate=SAMPLE_RATE,
            num_threads=1,
            provider="cpu",
            debug=False,
        )
        return sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=60.0,
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
