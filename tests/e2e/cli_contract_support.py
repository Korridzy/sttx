from __future__ import annotations

import socket
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

import sttx.cli as cli
from sttx.asr import ProgressCallback
from sttx.audio import PreparedAudio, normalize_media
from sttx.asr_events import ActivityCallback
from sttx.cli import RunnerDependencies
from sttx.model import ModelBundle
from sttx.output import OutputPaths, Segment, Transcript

SAMPLE_RATE: Final = 16_000
TEXT: Final = "offline media contract"


@dataclass(frozen=True, slots=True)
class FakeRecognizer:
    bundle: ModelBundle


@dataclass(frozen=True, slots=True)
class FakeVad:
    bundle: ModelBundle


@dataclass(slots=True)  # noqa: MUTABLE_OK
class GuardCounters:
    socket_attempts: int = 0
    real_recognizers: int = 0
    real_vads: int = 0
    injected_recognizers: int = 0
    injected_vads: int = 0


@dataclass(frozen=True, slots=True)
class FormatCase:
    suffix: str
    outdir: Path | None
    name: str | None
    expected_json: Path
    expected_txt: Path


@dataclass(frozen=True, slots=True)
class RunContext:
    media: Path
    case: FormatCase
    cwd: Path
    bundle: ModelBundle
    counters: GuardCounters
    silence: bool


@dataclass(frozen=True, slots=True)
class DecodeContext:
    media: Path
    tmp_path: Path
    bundle: ModelBundle
    counters: GuardCounters


@dataclass(frozen=True, slots=True)
class RunResult:
    exit_code: int
    json_path: Path | None
    txt_path: Path | None
    prepared_path: Path | None
    prepared_header: tuple[int, int, int, int] | None


@dataclass(frozen=True, slots=True)
class ErrorResult:
    exit_code: int
    stderr: str


def _transcribe_success(
    _audio: PreparedAudio,
    *,
    recognizer: FakeRecognizer,
    vad: FakeVad,
    progress: ProgressCallback | None = None,
    activity: ActivityCallback | None = None,
) -> Transcript:
    del recognizer, vad, progress, activity
    return Transcript(
        language="en",
        duration=1.0,
        segments=(Segment(0, 0.0, 1.0, TEXT),),
    )


def install_runtime_guards(
    monkeypatch: pytest.MonkeyPatch,
    counters: GuardCounters,
) -> None:
    def fail_socket(
        family: int = socket.AF_INET,
        kind: int = socket.SOCK_STREAM,
        proto: int = 0,
        fileno: int | None = None,
    ) -> socket.socket:
        del family, kind, proto, fileno
        counters.socket_attempts += 1
        raise AssertionError("socket construction is forbidden in offline CLI tests")

    def fail_recognizer(**kwargs: str | int | float) -> FakeRecognizer:
        del kwargs
        counters.real_recognizers += 1
        raise AssertionError("real recognizer construction is forbidden")

    def fail_vad(config: str | int | float, *, buffer_size_in_seconds: float) -> FakeVad:
        del config, buffer_size_in_seconds
        counters.real_vads += 1
        raise AssertionError("real VAD construction is forbidden")

    monkeypatch.setattr(socket, "socket", fail_socket)
    monkeypatch.setattr(cli.sherpa_onnx.OfflineRecognizer, "from_transducer", fail_recognizer)
    monkeypatch.setattr(cli.sherpa_onnx, "VoiceActivityDetector", fail_vad)


def bundle_from_tmp(tmp_path: Path) -> ModelBundle:
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    paths = {
        "encoder": model_dir / "encoder.int8.onnx",
        "decoder": model_dir / "decoder.int8.onnx",
        "joiner": model_dir / "joiner.int8.onnx",
        "tokens": model_dir / "tokens.txt",
        "silero": model_dir / "silero_vad.onnx",
    }
    for path in paths.values():
        path.write_bytes(b"offline-contract")
    return ModelBundle(
        encoder=paths["encoder"],
        decoder=paths["decoder"],
        joiner=paths["joiner"],
        tokens=paths["tokens"],
        silero=paths["silero"],
    )


def run_success(context: RunContext) -> RunResult:
    paths = expected_paths(context.cwd, context.case)
    seen_prepared: list[PreparedAudio] = []
    seen_header: list[tuple[int, int, int, int]] = []

    def make_recognizer(model_bundle: ModelBundle) -> FakeRecognizer:
        context.counters.injected_recognizers += 1
        return FakeRecognizer(model_bundle)

    def make_vad(model_bundle: ModelBundle) -> FakeVad:
        context.counters.injected_vads += 1
        return FakeVad(model_bundle)

    def transcribe(
        prepared: PreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: ProgressCallback | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del progress, activity
        assert recognizer.bundle == context.bundle
        assert vad.bundle == context.bundle
        seen_prepared.append(prepared)
        assert prepared.path.suffix == ".wav"
        assert prepared.path != context.media
        header = wav_header(prepared.path)
        seen_header.append(header)
        assert header[:3] == (SAMPLE_RATE, 1, 2)
        if context.silence:
            return Transcript(language="auto", duration=1.0, segments=())
        return Transcript(
            language="en",
            duration=1.0,
            segments=(Segment(id=0, start=0.0, end=1.0, text=TEXT),),
        )

    argv = [str(context.media), "--model-dir", str(context.cwd)]
    if context.case.outdir is not None:
        argv.extend(("--outdir", str(context.case.outdir)))
    if context.case.name is not None:
        argv.extend(("--output", context.case.name))
    exit_code = cli.run(
        argv,
        _dependencies=RunnerDependencies(
            normalize_media=normalize_media,
            resolve_bundle=lambda _model_dir: context.bundle,
            make_recognizer=make_recognizer,
            make_vad=make_vad,
            transcribe=transcribe,
        ),
        _cwd=context.cwd,
    )
    prepared = seen_prepared[0] if seen_prepared else None
    return RunResult(
        exit_code=exit_code,
        json_path=paths.json_path if paths.json_path.exists() else None,
        txt_path=paths.txt_path if paths.txt_path.exists() else None,
        prepared_path=prepared.path if prepared else None,
        prepared_header=seen_header[0] if seen_header else None,
    )


def run_error(
    argv: Sequence[str],
    capsys: pytest.CaptureFixture[str],
    bundle: ModelBundle,
) -> ErrorResult:
    exit_code = cli.run(
        argv,
        _dependencies=RunnerDependencies(
            normalize_media=normalize_media,
            resolve_bundle=lambda _model_dir: bundle,
            make_recognizer=FakeRecognizer,
            make_vad=FakeVad,
            transcribe=_transcribe_success,
        ),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    return ErrorResult(exit_code=exit_code, stderr=captured.err)


def run_decode_failure(context: DecodeContext) -> int:
    def make_recognizer(model_bundle: ModelBundle) -> FakeRecognizer:
        context.counters.injected_recognizers += 1
        return FakeRecognizer(model_bundle)

    def make_vad(model_bundle: ModelBundle) -> FakeVad:
        context.counters.injected_vads += 1
        return FakeVad(model_bundle)

    def fail_transcribe(
        prepared: PreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: ProgressCallback | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del prepared, recognizer, vad, progress, activity
        raise RuntimeError("injected decoder failure")

    return cli.run(
        [str(context.media), "--model-dir", str(context.tmp_path)],
        _dependencies=RunnerDependencies(
            normalize_media=normalize_media,
            resolve_bundle=lambda _model_dir: context.bundle,
            make_recognizer=make_recognizer,
            make_vad=make_vad,
            transcribe=fail_transcribe,
        ),
        _cwd=context.tmp_path,
    )


def expected_paths(cwd: Path, case: FormatCase) -> OutputPaths:
    json_path = cwd / case.expected_json if not case.expected_json.is_absolute() else case.expected_json
    txt_path = cwd / case.expected_txt if not case.expected_txt.is_absolute() else case.expected_txt
    return OutputPaths(json_path=json_path, txt_path=txt_path)


def expected_payload(text: str) -> dict[str, str | float | list[dict[str, str | float | int]]]:
    segments: list[dict[str, str | float | int]] = []
    language = "auto"
    if text:
        language = "en"
        segments.append({"id": 0, "start": 0.0, "end": 1.0, "text": text})
    return {
        "task": "transcribe",
        "language": language,
        "duration": 1.0,
        "text": text,
        "segments": segments,
    }


def staging_files(path: Path) -> list[Path]:
    return sorted(path.glob(".*.tmp"))


def wav_header(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as wav_file:
        return (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getnframes(),
        )
