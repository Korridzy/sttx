from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final, TypedDict

import pytest

from .cli_contract_support import (
    TEXT,
    DecodeContext,
    FormatCase,
    GuardCounters,
    RunContext,
    bundle_from_tmp,
    expected_payload,
    expected_paths,
    install_runtime_guards,
    run_decode_failure,
    run_error,
    run_success,
    staging_files,
)
from sttx.model import ModelBundle

FFMPEG_TIMEOUT: Final = 10.0


class TranscriptPayload(TypedDict):
    task: str
    language: str
    duration: float
    text: str
    segments: list[dict[str, int | float | str]]


@pytest.fixture(autouse=True)
def guarded_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[GuardCounters]:
    counters = GuardCounters()
    install_runtime_guards(monkeypatch, counters)
    yield counters
    assert counters.socket_attempts == 0
    assert counters.real_recognizers == 0
    assert counters.real_vads == 0


@pytest.fixture
def bundle(tmp_path: Path) -> ModelBundle:
    return bundle_from_tmp(tmp_path)


def test_formats_names_overwrite_schema_streams_and_cleanup(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    guarded_runtime: GuardCounters,
    bundle: ModelBundle,
) -> None:
    cases = (
        FormatCase(".wav", None, None, Path("transcriptions/clip.json"), Path("transcriptions/clip.txt")),
        FormatCase(".mp3", Path("nested/results"), None, Path("nested/results/clip.json"), Path("nested/results/clip.txt")),
        FormatCase(".ogg", tmp_path / "absolute-results", None, Path("absolute-results/archive.tar.json"), Path("absolute-results/archive.tar.txt")),
        FormatCase(".mp4", None, "episode.one", Path("transcriptions/episode.one.json"), Path("transcriptions/episode.one.txt")),
    )
    for case in cases:
        media = generate_media(tmp_path, case)
        paths = expected_paths(tmp_path, case)
        paths.json_path.parent.mkdir(parents=True, exist_ok=True)
        paths.json_path.write_text("stale json", encoding="utf-8")
        paths.txt_path.write_text("stale txt", encoding="utf-8")

        result = run_success(
            RunContext(
                media=media,
                case=case,
                cwd=tmp_path,
                bundle=bundle,
                counters=guarded_runtime,
                silence=False,
            )
        )

        captured = capsys.readouterr()
        assert result.exit_code == 0
        assert result.json_path == paths.json_path
        assert result.txt_path == paths.txt_path
        assert captured.out == f"{paths.json_path}\n{paths.txt_path}\n"
        assert captured.err == ""
        assert result.prepared_header is not None
        assert result.prepared_header[:3] == (16_000, 1, 2)
        assert result.prepared_header[3] > 0
        assert result.prepared_path is not None
        assert not result.prepared_path.exists()
        payload = read_payload(paths.json_path)
        assert payload == expected_payload(TEXT)
        assert paths.txt_path.read_text(encoding="utf-8") == payload["text"]
        assert "stale" not in paths.json_path.read_text(encoding="utf-8")
        assert "stale" not in paths.txt_path.read_text(encoding="utf-8")
        assert staging_files(paths.json_path.parent) == []

    assert guarded_runtime.injected_recognizers == len(cases)
    assert guarded_runtime.injected_vads == len(cases)


def test_error_and_silence_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    guarded_runtime: GuardCounters,
    bundle: ModelBundle,
) -> None:
    media = generate_media(
        tmp_path,
        FormatCase(".wav", None, None, Path("transcriptions/clip.json"), Path("transcriptions/clip.txt")),
    )

    error_args = ["--model-dir", str(tmp_path), "--outdir", str(tmp_path / "errors")]
    missing_path = tmp_path / "missing.wav"
    missing = run_error([str(missing_path), *error_args], capsys, bundle)
    assert missing.exit_code == 1
    assert missing.stderr == f"error: cannot prepare audio from {missing_path}: input is not a readable file\n"

    unreadable = tmp_path / "unreadable.wav"
    unreadable.write_bytes(b"not readable")
    unreadable.chmod(0o000)
    with monkeypatch.context() as unreadable_patch:
        unreadable_patch.setattr("sttx.cli.os.access", lambda _path, _mode: False)
        unreadable_result = run_error([str(unreadable), *error_args], capsys, bundle)
    unreadable.chmod(0o600)
    assert unreadable_result.exit_code == 1
    assert unreadable_result.stderr == f"error: cannot prepare audio from {unreadable}: input is not a readable file\n"

    invalid_name = run_error([str(media), "--output-name", "bad.json", *error_args], capsys, bundle)
    assert invalid_name.exit_code == 1
    assert invalid_name.stderr == "error: invalid output path bad.json: name must be one stem without suffix\n"

    monkeypatch.setattr("shutil.which", lambda _command: None)
    missing_ffmpeg = run_error([str(media), *error_args], capsys, bundle)
    assert missing_ffmpeg.exit_code == 1
    assert missing_ffmpeg.stderr == (
        f"error: cannot prepare audio from {media}: ffmpeg is not available\n"
        "sttx uses ffmpeg to prepare media for transcription. Install ffmpeg "
        "and make it available on PATH: https://ffmpeg.org/download.html\n"
    )
    monkeypatch.undo()

    no_audio = generate_no_audio_mp4(tmp_path)
    no_audio_result = run_error([str(no_audio), *error_args], capsys, bundle)
    assert no_audio_result.exit_code == 1
    assert no_audio_result.stderr.startswith(f"error: cannot prepare audio from {no_audio}: ffmpeg did not produce audio")

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    blocked_result = run_error(
        [str(media), "--outdir", str(blocked), "--model-dir", str(tmp_path)],
        capsys,
        bundle,
    )
    assert blocked_result.exit_code == 1
    assert blocked_result.stderr == f"error: invalid output path {blocked}: [Errno 17] File exists: '{blocked}'\n"

    unwritable = tmp_path / "unwritable"
    unwritable.mkdir()
    with monkeypatch.context() as output_patch:
        output_patch.setattr(
            "sttx.output.tempfile.NamedTemporaryFile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError(13, "Permission denied")),
        )
        output_result = run_error([str(media), "--outdir", str(unwritable), "--model-dir", str(tmp_path)], capsys, bundle)
    assert output_result.exit_code == 1
    assert output_result.stderr.startswith(f"error: failed to write output {unwritable / 'clip.json'}: [Errno 13] Permission denied")
    assert staging_files(unwritable) == []

    decode_context = DecodeContext(media, tmp_path, bundle, guarded_runtime)
    assert run_decode_failure(decode_context) == 2
    decode_captured = capsys.readouterr()
    assert decode_captured.out == ""
    assert "transcription decode failed" in decode_captured.err
    assert "Traceback" not in decode_captured.err

    silence_case = FormatCase(
        ".wav",
        Path("silence"),
        "quiet",
        Path("silence/quiet.json"),
        Path("silence/quiet.txt"),
    )
    silence = run_success(
        RunContext(media, silence_case, tmp_path, bundle, guarded_runtime, True)
    )
    silence_captured = capsys.readouterr()
    assert silence.exit_code == 0
    assert silence.json_path is not None
    assert silence.txt_path is not None
    assert silence_captured.out == f"{silence.json_path}\n{silence.txt_path}\n"
    assert "warning: no speech detected" in silence_captured.err
    assert read_payload(silence.json_path) == expected_payload("")
    assert silence.txt_path.read_text(encoding="utf-8") == ""
    assert staging_files(silence.json_path.parent) == []


def generate_media(tmp_path: Path, case: FormatCase) -> Path:
    stem = "archive.tar" if case.suffix == ".ogg" else "clip"
    path = tmp_path / f"{stem}{case.suffix}"
    run_ffmpeg(media_command(case.suffix, path))
    return path


def media_command(suffix: str, path: Path) -> list[str]:
    ffmpeg = ffmpeg_path()
    common = [ffmpeg, "-f", "lavfi", "-i", "sine=frequency=440:duration=0.1"]
    match suffix:
        case ".wav":
            return [*common, "-ac", "1", "-ar", "16000", "-y", str(path)]
        case ".mp3":
            return [*common, "-codec:a", "libmp3lame", "-y", str(path)]
        case ".ogg":
            return [*common, "-codec:a", "libvorbis", "-y", str(path)]
        case ".mp4":
            return [
                *common,
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:d=0.1",
                "-shortest",
                "-c:a",
                "aac",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(path),
            ]
        case unmatched:
            raise AssertionError(f"unsupported fixture suffix {unmatched}")


def generate_no_audio_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "video-only.mp4"
    run_ffmpeg(
        [
            ffmpeg_path(),
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ]
    )
    return path


def ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    return ffmpeg


def run_ffmpeg(command: Sequence[str]) -> None:
    completed = subprocess.run(
        [*command[:1], "-nostdin", "-hide_banner", "-loglevel", "error", *command[1:]],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=FFMPEG_TIMEOUT,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


def read_payload(path: Path) -> TranscriptPayload:
    return json.loads(path.read_text(encoding="utf-8"))
