from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from sttx.audio import (
    AudioCancellationError,
    AudioEnvironmentError,
    PreparedAudio,
    normalize_media,
)

SAMPLE_RATE = 16_000
FFMPEG_GENERATE_PREFIX = (
    "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"
)


def write_wav(path: Path, sample_count: int = 320) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(b"\x01\x00" * sample_count)


def generate_media(ffmpeg: str, path: Path, source: str) -> None:
    subprocess.run(
        [ffmpeg, *FFMPEG_GENERATE_PREFIX, source, "-y", str(path)],
        check=True,
    )


class SuccessfulPopen:
    def __init__(self, argv: Sequence[str], sample_count: int = 320) -> None:
        self.argv = tuple(argv)
        self.sample_count = sample_count
        self.returncode: int | None = None
        self.pid = 8_001

    def communicate(self) -> tuple[None, bytes]:
        write_wav(Path(self.argv[-1]), self.sample_count)
        self.returncode = 0
        return None, b""

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0


def install_successful_popen(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[SuccessfulPopen],
    sample_count: int = 320,
) -> None:
    def start(
        argv: Sequence[str],
        *,
        start_new_session: bool,
        stdout: int,
        stderr: int,
    ) -> SuccessfulPopen:
        assert start_new_session is True
        assert stdout == subprocess.DEVNULL
        assert stderr == subprocess.PIPE
        process = SuccessfulPopen(argv, sample_count)
        calls.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(shutil, "which", lambda _command: "ffmpeg")


def test_ffmpeg_argv_is_16k_mono_pcm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: readable media and a fake process that emits a valid normalized WAV.
    source = tmp_path / "input.mp4"
    source.write_bytes(b"media")
    calls: list[SuccessfulPopen] = []
    install_successful_popen(monkeypatch, calls)

    output_path: Path | None = None

    # When: the media is normalized.
    with normalize_media(source) as prepared:
        output_path = prepared.path

    # Then: ffmpeg receives the single exact conversion path without a shell.
    assert output_path is not None
    assert calls[0].argv == (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output_path),
    )
    assert not output_path.exists()


def test_wav_input_still_uses_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an input that is already a valid 16 kHz mono WAV.
    source = tmp_path / "already.wav"
    write_wav(source)
    calls: list[SuccessfulPopen] = []
    install_successful_popen(monkeypatch, calls)

    # When: it is normalized.
    with normalize_media(source):
        pass

    # Then: the WAV follows the same ffmpeg process path.
    assert len(calls) == 1
    assert calls[0].argv[6] == str(source)


def test_duration_uses_sample_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: ffmpeg produces exactly 48,001 samples.
    source = tmp_path / "duration.media"
    source.write_bytes(b"media")
    calls: list[SuccessfulPopen] = []
    install_successful_popen(monkeypatch, calls, sample_count=48_001)

    # When: normalized audio metadata is read.
    with normalize_media(source) as prepared:
        # Then: duration is derived from sample count and the target rate.
        assert prepared.sample_count == 48_001
        assert prepared.duration == 48_001 / SAMPLE_RATE


def test_missing_ffmpeg_and_no_audio_are_environment_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a readable input but no ffmpeg on PATH.
    source = tmp_path / "input.media"
    source.write_bytes(b"media")
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    # When/Then: dependency absence is classified for the future CLI boundary.
    with pytest.raises(AudioEnvironmentError):
        normalize_media(source)

    # Given: real video-only media.
    monkeypatch.undo()
    video_only = tmp_path / "video-only.mkv"
    no_audio_output = tmp_path / "no-audio.wav"
    monkeypatch.setattr(
        tempfile,
        "mkstemp",
        lambda **_kwargs: (
            os.open(no_audio_output, os.O_CREAT | os.O_RDWR | os.O_TRUNC),
            str(no_audio_output),
        ),
    )
    generate_media(ffmpeg, video_only, "color=c=black:s=16x16:d=0.1")

    # When/Then: media without an audio track is the same classified environment error.
    with pytest.raises(AudioEnvironmentError):
        normalize_media(video_only)
    assert not no_audio_output.exists()


def test_cleanup_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: ffmpeg reports success but leaves malformed output at a known temp path.
    source = tmp_path / "input.media"
    source.write_bytes(b"media")
    output = tmp_path / "partial.wav"
    calls: list[SuccessfulPopen] = []
    install_successful_popen(monkeypatch, calls)

    def malformed_communicate(self: SuccessfulPopen) -> tuple[None, bytes]:
        output.write_bytes(b"not a wav")
        self.returncode = 0
        return None, b""

    monkeypatch.setattr(SuccessfulPopen, "communicate", malformed_communicate)
    monkeypatch.setattr(
        tempfile,
        "mkstemp",
        lambda **_kwargs: (os.open(output, os.O_CREAT), str(output)),
    )

    # When: produced output validation fails.
    with pytest.raises(AudioEnvironmentError):
        normalize_media(source)

    # Then: partial system-temp output is removed.
    assert not output.exists()


def test_real_ffmpeg_normalizes_non_wav_and_cleans_context(tmp_path: Path) -> None:
    # Given: real compressed audio generated by ffmpeg.
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "tone.mp3"
    generate_media(ffmpeg, source, "sine=frequency=440:duration=0.1")

    output: Path | None = None
    header: tuple[int, int, int, int] | None = None

    # When: production normalization runs.
    with normalize_media(source) as prepared:
        output = prepared.path
        with wave.open(str(output), "rb") as wav_file:
            header = (
                wav_file.getframerate(),
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getnframes(),
            )

    # Then: the real artifact has the promised header and is context-cleaned.
    assert header is not None
    assert output is not None
    assert header[:3] == (SAMPLE_RATE, 1, 2)
    assert header[3] > 0
    assert not output.exists()


def test_sleeping_ffmpeg_is_terminated_then_killed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a fake ffmpeg executable that ignores TERM and sleeps indefinitely.
    source = tmp_path / "input.media"
    source.write_bytes(b"media")
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    pid_file = tmp_path / "ffmpeg.pid"
    temp_file = tmp_path / "ffmpeg-temp.path"
    wrapper = wrapper_dir / "ffmpeg"
    wrapper.write_text(
        "#!/bin/sh\n"
        "trap '' TERM\n"
        f"printf '%s' \"$$\" > '{pid_file}'\n"
        f"printf '%s' \"$15\" > '{temp_file}'\n"
        "sleep 60\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrapper_dir}:{os.environ['PATH']}")

    def interrupt_when_started() -> None:
        deadline = time.monotonic() + 3.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_when_started)
    interrupter.start()

    # When: cancellation interrupts the blocking ffmpeg conversion.
    with pytest.raises(AudioCancellationError):
        normalize_media(source)
    interrupter.join(timeout=3.0)

    # Then: TERM escalation reaches KILL, the group is dead, and temp output is absent.
    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    output = Path(temp_file.read_text(encoding="utf-8"))
    assert not output.exists()


def test_prepared_audio_cleanup_is_idempotent(tmp_path: Path) -> None:
    # Given: a prepared temporary WAV resource.
    output = tmp_path / "prepared.wav"
    write_wav(output)
    prepared = PreparedAudio(path=output, sample_count=320)

    # When: cleanup is repeated as after repeated interruption handling.
    prepared.cleanup()
    prepared.cleanup()

    # Then: the resource remains absent without an error.
    assert not output.exists()


def test_missing_and_unreadable_inputs_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a missing media path.
    missing = tmp_path / "missing.media"

    # When/Then: normalization rejects it before allocating a temporary WAV.
    with pytest.raises(AudioEnvironmentError):
        normalize_media(missing)

    # Given: an existing path that fails the readability boundary check.
    unreadable = tmp_path / "unreadable.media"
    unreadable.write_bytes(b"media")
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)

    # When/Then: unreadability is a classified environment failure.
    with pytest.raises(AudioEnvironmentError):
        normalize_media(unreadable)
