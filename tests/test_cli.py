from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType

import pytest

from sttx.audio import AudioEnvironmentError
from sttx.asr_events import (
    ActivityCallback,
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
from sttx.model import ModelBundle, ModelEnvironmentError
from sttx.output import OutputPathError, OutputPaths, Segment, Transcript, write_outputs
from sttx.cli import RunnerDependencies


@dataclass(frozen=True, slots=True)
class FakeRecognizer:
    bundle: ModelBundle


@dataclass(frozen=True, slots=True)
class FakeVad:
    bundle: ModelBundle


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakePreparedAudio:
    path: Path
    sample_count: int
    cleaned: bool = False
    cleanup_seen_after_transcription: bool = False

    @property
    def duration(self) -> float:
        return self.sample_count / 16_000

    def __enter__(self) -> FakePreparedAudio:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.cleaned = True
        return False


def _bundle(tmp_path: Path) -> ModelBundle:
    return ModelBundle(
        encoder=tmp_path / "encoder.int8.onnx",
        decoder=tmp_path / "decoder.int8.onnx",
        joiner=tmp_path / "joiner.int8.onnx",
        tokens=tmp_path / "tokens.txt",
        silero=tmp_path / "silero_vad.onnx",
    )


def _transcript(text: str = "Привет мир") -> Transcript:
    return Transcript(
        language="ru",
        duration=1.25,
        segments=(Segment(id=0, start=0.0, end=1.25, text=text),) if text else (),
    )


def _fake_transcribe(
    _audio: FakePreparedAudio,
    *,
    recognizer: FakeRecognizer,
    vad: FakeVad,
    progress: Callable[[int, int], None] | None = None,
    activity: ActivityCallback | None = None,
) -> Transcript:
    del recognizer, vad, progress, activity
    return _transcript()


def _fake_dependencies(
    tmp_path: Path,
) -> RunnerDependencies[FakePreparedAudio, FakeRecognizer, FakeVad]:
    return RunnerDependencies(
        normalize_media=lambda _path: FakePreparedAudio(
            path=tmp_path / "prepared.wav",
            sample_count=16_000,
        ),
        resolve_bundle=lambda _model_dir: _bundle(tmp_path),
        make_recognizer=FakeRecognizer,
        make_vad=FakeVad,
        transcribe=_fake_transcribe,
    )


def test_parser_contract() -> None:
    # Given: the public parser builder.
    from sttx.cli import build_parser

    # When: supported arguments are parsed.
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "media.mp4",
            "--output",
            "episode",
            "--outdir",
            "results",
            "--model-dir",
            "models",
            "--verbose",
        ]
    )

    # Then: the namespace carries only the public transcription controls.
    assert namespace.media == Path("media.mp4")
    assert namespace.output == "episode"
    assert namespace.outdir == Path("results")
    assert namespace.model_dir == Path("models")
    assert namespace.verbose == 1
    assert parser.parse_args(["media.mp4", "-vv"]).verbose == 2
    assert parser.parse_args(["media.mp4"]).debug is False
    assert parser.parse_args(["media.mp4", "--debug"]).debug is True
    assert parser.parse_args(["media.mp4"]).log_format == "text"
    assert parser.parse_args(["media.mp4", "--log-format", "json"]).log_format == "json"


def test_output_precedence_and_exact_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: readable media, explicit output controls, and a path seam that records inputs.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    calls: list[tuple[Path, Path | None, str | None, Path]] = []
    paths = OutputPaths(json_path=tmp_path / "exact.json", txt_path=tmp_path / "exact.txt")

    def capture_paths(
        input_path: Path,
        outdir: Path | None,
        name: str | None,
        cwd: Path,
    ) -> OutputPaths:
        calls.append((input_path, outdir, name, cwd))
        return paths

    # When: the runner resolves output paths before acquiring models.
    exit_code = run(
        [str(media), "-o", "episode", "-d", "relative", "--model-dir", str(tmp_path)],
        _dependencies=_fake_dependencies(tmp_path),
        _output_paths=capture_paths,
        _write_outputs=lambda _transcript, _paths: None,
        _cwd=tmp_path,
    )

    # Then: CLI options reach the Todo 2 naming/outdir boundary exactly.
    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [(media, Path("relative"), "episode", tmp_path)]
    assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"


def test_stdout_contains_only_final_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: injected seams that record the CLI boundary order and resource lifetime.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    bundle = _bundle(tmp_path)
    prepared = FakePreparedAudio(path=tmp_path / "prepared.wav", sample_count=20_000)
    events: list[str] = []
    paths = OutputPaths(json_path=tmp_path / "episode.json", txt_path=tmp_path / "episode.txt")

    def normalize(path: Path) -> FakePreparedAudio:
        events.append(f"normalize:{path.name}")
        return prepared

    def resolve(model_dir: Path | None) -> ModelBundle:
        events.append(f"resolve:{model_dir}")
        return bundle

    def make_recognizer(model_bundle: ModelBundle) -> FakeRecognizer:
        events.append("recognizer")
        return FakeRecognizer(model_bundle)

    def make_vad(model_bundle: ModelBundle) -> FakeVad:
        events.append("vad")
        return FakeVad(model_bundle)

    def transcribe(
        audio: FakePreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: Callable[[int, int], None] | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del recognizer, vad, progress, activity
        assert audio is prepared
        prepared.cleanup_seen_after_transcription = prepared.cleaned
        events.append("transcribe")
        return _transcript()

    def output_paths(
        input_path: Path,
        outdir: Path | None,
        name: str | None,
        cwd: Path,
    ) -> OutputPaths:
        events.append(f"paths:{input_path.name}:{outdir}:{name}:{cwd}")
        return paths

    def write_outputs(transcript: Transcript, output: OutputPaths) -> None:
        events.append(f"write:{transcript.text}")
        assert output == paths

    dependencies = RunnerDependencies(
        normalize_media=normalize,
        resolve_bundle=resolve,
        make_recognizer=make_recognizer,
        make_vad=make_vad,
        transcribe=transcribe,
    )

    # When: the dependency-injectable runner completes.
    exit_code = run(
        [str(media), "-o", "episode", "-d", str(tmp_path), "--model-dir", str(tmp_path)],
        _dependencies=dependencies,
        _output_paths=output_paths,
        _write_outputs=write_outputs,
        _cwd=tmp_path,
    )

    # Then: stdout contains exactly the final artifact paths and stderr stays quiet.
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"
    assert captured.err == ""
    assert events == [
        f"paths:{media.name}:{tmp_path}:episode:{tmp_path}",
        f"normalize:{media.name}",
        f"resolve:{tmp_path}",
        "recognizer",
        "vad",
        "transcribe",
        "write:Привет мир",
    ]
    assert prepared.cleanup_seen_after_transcription is False
    assert prepared.cleaned is True


def test_verbose_prints_progress_to_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a successful injected run with verbose output requested.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    paths = OutputPaths(json_path=tmp_path / "episode.json", txt_path=tmp_path / "episode.txt")

    # When: verbose mode is enabled.
    exit_code = run(
        [str(media), "--verbose", "--model-dir", str(tmp_path)],
        _dependencies=_fake_dependencies(tmp_path),
        _output_paths=lambda _input_path, _outdir, _name, _cwd: paths,
        _write_outputs=lambda _transcript, _paths: None,
        _cwd=tmp_path,
    )

    # Then: final paths remain on stdout while progress uses stderr.
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"
    assert f"input path={media}" in captured.err
    assert f"output json={paths.json_path} txt={paths.txt_path}" in captured.err
    assert "normalize start" in captured.err
    assert "normalize complete duration=1.00s" in captured.err
    assert "models resolve start" in captured.err
    assert "models resolve complete" in captured.err
    assert "recognizer initialize start" in captured.err
    assert "recognizer initialize complete" in captured.err
    assert "VAD initialize start" in captured.err
    assert "VAD initialize complete" in captured.err
    assert "transcribe start" in captured.err
    assert "transcribe complete language=ru segments=1" in captured.err
    assert "write outputs start" in captured.err
    assert "write outputs complete" in captured.err
    assert "transcribe progress=" not in captured.err
    assert "complete duration=1.25s rtf=" in captured.err


def test_double_verbose_prints_transcription_progress_to_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    paths = OutputPaths(json_path=tmp_path / "episode.json", txt_path=tmp_path / "episode.txt")

    def transcribe(
        _audio: FakePreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: Callable[[int, int], None] | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del recognizer, vad, activity
        assert progress is not None
        progress(8_000, 16_000)
        progress(16_000, 16_000)
        return _transcript()

    exit_code = run(
        [str(media), "-vv", "--model-dir", str(tmp_path)],
        _dependencies=replace(_fake_dependencies(tmp_path), transcribe=transcribe),
        _output_paths=lambda _input_path, _outdir, _name, _cwd: paths,
        _write_outputs=lambda _transcript, _paths: None,
        _cwd=tmp_path,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"
    assert "transcribe progress=50% audio=0.50s/1.00s" in captured.err
    assert "transcribe progress=100% audio=1.00s/1.00s" in captured.err
    assert "eta=0.00s" in captured.err


def test_json_log_format_emits_structured_progress_and_preserves_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a transcriber that reports deterministic live progress.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    paths = OutputPaths(json_path=tmp_path / "episode.json", txt_path=tmp_path / "episode.txt")

    def transcribe(
        _audio: FakePreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: Callable[[int, int], None] | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del recognizer, vad, activity
        assert progress is not None
        progress(8_000, 16_000)
        progress(16_000, 16_000)
        return _transcript()

    # When: JSON Lines logging is requested with live transcription progress.
    exit_code = run(
        [str(media), "-vv", "--log-format", "json", "--model-dir", str(tmp_path)],
        _dependencies=replace(_fake_dependencies(tmp_path), transcribe=transcribe),
        _output_paths=lambda _input_path, _outdir, _name, _cwd: paths,
        _write_outputs=lambda _transcript, _paths: None,
        _cwd=tmp_path,
    )

    # Then: stderr is machine-readable telemetry and stdout remains the artifact contract.
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert all(isinstance(record["event"], str) for record in records)
    assert all(isinstance(record["elapsed_seconds"], float) for record in records)
    progress_records = [
        record for record in records if record["event"] == "transcribe_progress"
    ]
    assert [record["percent"] for record in progress_records] == [50, 100]
    assert progress_records[0]["audio_seconds"] == 0.5
    assert progress_records[0]["audio_total_seconds"] == 1.0
    assert progress_records[0]["eta_seconds"] >= 0.0
    assert "Привет мир" not in captured.err


def test_json_log_format_emits_structured_validation_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a missing media path and JSON log formatting.
    from sttx.cli import run

    media = tmp_path / "missing.mp4"

    # When: validation fails before the transcription pipeline starts.
    exit_code = run([str(media), "--log-format", "json"])

    # Then: the failure remains a single actionable JSON record on stderr.
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "error"
    assert records[0]["stage"] == "validate"
    assert records[0]["exception_type"] == "AudioEnvironmentError"
    assert "input is not a readable file" in records[0]["message"]


def test_debug_prints_internal_diagnostics_to_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a successful injected run with deep diagnostics requested.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    prepared_path = tmp_path / "prepared.wav"
    paths = OutputPaths(json_path=tmp_path / "episode.json", txt_path=tmp_path / "episode.txt")
    activity_calls: list[bool] = []

    def transcribe_with_activity(
        _audio: FakePreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: Callable[[int, int], None] | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del recognizer, vad, progress
        assert activity is not None
        activity(ScanStarted(total_samples=16_000))
        activity(ScanAdvanced(processed_samples=8_000, total_samples=16_000))
        activity(VadSegmentReady(index=1, start_sample=0, sample_count=8_000))
        activity(DecodeStarted(index=1, start_sample=0, sample_count=8_000))
        activity(DecodeFinished(index=1, start_sample=0, sample_count=8_000))
        activity(LanguageReported(language="ru"))
        activity(WordCountUpdated(word_count=2))
        activity(ScanFinished(total_samples=16_000))
        activity(
            TranscriptionSummary(
                total_samples=16_000,
                voiced_samples=8_000,
                vad_segments=1,
                decoded_samples=8_000,
                decode_chunks=1,
                word_count=2,
                language="ru",
                transcript_segments=1,
            )
        )
        activity_calls.append(True)
        return _transcript()

    # When: debug mode is enabled.
    exit_code = run(
        [str(media), "--debug", "--model-dir", str(tmp_path)],
        _dependencies=replace(
            _fake_dependencies(tmp_path),
            normalize_media=lambda _path: FakePreparedAudio(
                path=prepared_path,
                sample_count=16_000,
            ),
            transcribe=transcribe_with_activity,
        ),
        _output_paths=lambda _input_path, _outdir, _name, _cwd: paths,
        _write_outputs=lambda _transcript, _paths: None,
        _cwd=tmp_path,
    )

    # Then: diagnostics are complete enough for reproduction but stdout is unchanged.
    captured = capsys.readouterr()
    assert exit_code == 0
    assert activity_calls == [True]
    assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"
    assert "debug configuration" in captured.err
    assert "debug ffmpeg argv=ffmpeg -nostdin" in captured.err
    assert f"debug normalized_wav={prepared_path}" in captured.err
    assert "debug model source=explicit" in captured.err
    assert "debug model asset=encoder path=" in captured.err
    assert "debug stage=normalize duration=" in captured.err
    assert "debug asr scan progress=50% audio=0.50s/1.00s" in captured.err
    assert "debug asr vad segment=1 audio=0.00s+0.50s" in captured.err
    assert "debug asr decode start chunk=1" in captured.err
    assert "debug asr decode done chunk=1" in captured.err
    assert "debug asr language reported=ru" in captured.err
    assert "debug asr words=2" in captured.err
    assert "debug asr summary vad_segments=1 decode_chunks=1 words=2" in captured.err


def test_debug_prints_traceback_for_unexpected_write_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a completed transcription whose output seam raises an unexpected exception.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    paths = OutputPaths(json_path=tmp_path / "episode.json", txt_path=tmp_path / "episode.txt")

    def broken_write(_transcript: Transcript, _paths: OutputPaths) -> None:
        raise KeyError("staging metadata missing")

    # When: debug mode handles the failure boundary.
    exit_code = run(
        [str(media), "--debug", "--model-dir", str(tmp_path)],
        _dependencies=_fake_dependencies(tmp_path),
        _output_paths=lambda _input_path, _outdir, _name, _cwd: paths,
        _write_outputs=broken_write,
        _cwd=tmp_path,
    )

    # Then: callers receive the normal runtime exit code plus debug context and traceback.
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "debug failure stage=write exception=KeyError" in captured.err
    assert "Traceback" in captured.err
    assert "KeyError: 'staging metadata missing'" in captured.err


def test_run_validates_input_and_output_before_bundle_acquisition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: an unreadable media path and output/model seams that must not run.
    from sttx.cli import run

    media = tmp_path / "missing.mp4"

    def forbidden_paths(
        input_path: Path,
        outdir: Path | None,
        name: str | None,
        cwd: Path,
    ) -> OutputPaths:
        del input_path, outdir, name, cwd
        raise AssertionError("output paths must not be resolved for unreadable input")

    def forbidden_resolve(model_dir: Path | None) -> ModelBundle:
        del model_dir
        raise AssertionError("bundle acquisition must not run before input validation")

    # When: the runner sees the missing input.
    exit_code = run(
        [str(media), "--output", "episode"],
        _dependencies=replace(
            _fake_dependencies(tmp_path),
            resolve_bundle=forbidden_resolve,
        ),
        _output_paths=forbidden_paths,
        _cwd=tmp_path,
    )

    # Then: it exits as an environment failure without raw traceback output.
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "input is not a readable file" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_output_names_exit_one_before_model_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: readable media and an invalid output stem.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")

    def forbidden_resolve(model_dir: Path | None) -> ModelBundle:
        del model_dir
        raise AssertionError("bundle acquisition must not run before output validation")

    # When: output validation fails.
    exit_code = run(
        [str(media), "--output", "bad.json"],
        _dependencies=replace(
            _fake_dependencies(tmp_path),
            resolve_bundle=forbidden_resolve,
        ),
        _cwd=tmp_path,
    )

    # Then: invalid output maps to the environment/configuration failure class.
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "invalid output path" in captured.err


def test_silence_warns_and_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: transcription completes with no segments.
    from sttx.cli import run

    media = tmp_path / "silent.wav"
    media.write_bytes(b"media")
    paths = OutputPaths(json_path=tmp_path / "silent.json", txt_path=tmp_path / "silent.txt")
    writes: list[Transcript] = []

    def transcribe_silence(
        _audio: FakePreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: Callable[[int, int], None] | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del recognizer, vad, progress, activity
        return _transcript(text="")

    # When: the runner transcribes silence.
    exit_code = run(
        [str(media), "--model-dir", str(tmp_path)],
        _dependencies=replace(
            _fake_dependencies(tmp_path),
            transcribe=transcribe_silence,
        ),
        _output_paths=lambda _input_path, _outdir, _name, _cwd: paths,
        _write_outputs=lambda transcript, _paths: writes.append(transcript),
        _cwd=tmp_path,
    )

    # Then: silence is a successful empty transcript with a warning on stderr.
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"
    assert "warning: no speech detected" in captured.err
    assert writes == [_transcript(text="")]


@pytest.mark.parametrize(
    "exception",
    [
        AudioEnvironmentError(Path("input.wav"), "ffmpeg is not available"),
        ModelEnvironmentError(Path("models"), "required file is missing"),
        OutputPathError(Path("bad"), "name must be one stem without suffix"),
    ],
)
def test_environment_error_table_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    exception: BaseException,
) -> None:
    # Given: a readable input and a seam raising a classified domain error.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")

    def raise_error(model_dir: Path | None) -> ModelBundle:
        del model_dir
        raise exception

    # When: the error reaches the CLI boundary.
    exit_code = run(
        [str(media), "--model-dir", str(tmp_path)],
        _dependencies=replace(_fake_dependencies(tmp_path), resolve_bundle=raise_error),
        _cwd=tmp_path,
    )

    # Then: it is rendered without a raw trace and with the documented code.
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert str(exception) in captured.err
    assert "Traceback" not in captured.err


def test_argparse_and_transcription_errors_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: the production runner.
    from sttx.cli import run
    from sttx.asr import TranscriptionError

    # When: syntax is invalid.
    invalid_code = run([])

    # Then: argparse syntax maps to code 2.
    invalid = capsys.readouterr()
    assert invalid_code == 2
    assert invalid.out == ""
    assert "usage:" in invalid.err

    # Given: readable media and injected runtime failures.
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")

    def raise_transcription(
        _audio: FakePreparedAudio,
        *,
        recognizer: FakeRecognizer,
        vad: FakeVad,
        progress: Callable[[int, int], None] | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del recognizer, vad, progress, activity
        raise TranscriptionError("decode failed")

    def raise_recognizer(model_bundle: ModelBundle) -> FakeRecognizer:
        del model_bundle
        raise argparse.ArgumentTypeError("recognizer construction failed")

    def raise_vad(model_bundle: ModelBundle) -> FakeVad:
        del model_bundle
        raise argparse.ArgumentTypeError("VAD construction failed")

    # When: recognizer construction, VAD construction, and decoding fail.
    recognizer_code = run(
        [str(media), "--model-dir", str(tmp_path)],
        _dependencies=replace(
            _fake_dependencies(tmp_path),
            make_recognizer=raise_recognizer,
        ),
        _cwd=tmp_path,
    )
    vad_code = run(
        [str(media), "--model-dir", str(tmp_path)],
        _dependencies=replace(_fake_dependencies(tmp_path), make_vad=raise_vad),
        _cwd=tmp_path,
    )
    transcription_code = run(
        [str(media), "--model-dir", str(tmp_path)],
        _dependencies=replace(
            _fake_dependencies(tmp_path),
            transcribe=raise_transcription,
        ),
        _cwd=tmp_path,
    )

    # Then: each runtime user-facing failure maps to code 2 without stdout.
    runtime = capsys.readouterr()
    assert recognizer_code == 2
    assert vad_code == 2
    assert transcription_code == 2
    assert runtime.out == ""
    assert "Traceback" not in runtime.err


def test_output_failure_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a write failure on the second atomic replace through the CLI.
    from sttx.cli import run

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"media")
    outdir = tmp_path / "out"
    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def fail_second_replace(source: str, destination: str) -> None:
        calls.append((source, destination))
        if len(calls) == 2:
            raise OSError("injected replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)

    # When: real output writing fails after staging files are created.
    exit_code = run(
        [str(media), "--output", "episode", "--outdir", str(outdir), "--model-dir", str(tmp_path)],
        _dependencies=_fake_dependencies(tmp_path),
        _write_outputs=write_outputs,
        _cwd=tmp_path,
    )

    # Then: the CLI maps output filesystem failure to 1 and leaves no staging files.
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "failed to write output" in captured.err
    assert not list(outdir.glob(".episode.*.tmp"))


def test_main_delegates_to_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a patched runner return code.
    import sttx.cli as cli

    monkeypatch.setattr(cli, "run", lambda argv=None: 17)

    # When/Then: main raises SystemExit with the runner result.
    with pytest.raises(SystemExit) as captured:
        cli.main()
    assert captured.value.code == 17
