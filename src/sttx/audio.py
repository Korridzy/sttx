from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Final

SAMPLE_RATE: Final = 16_000
SAMPLE_WIDTH: Final = 2
CHANNEL_COUNT: Final = 1
PROCESS_STOP_TIMEOUT: Final = 1.0


@dataclass(frozen=True, slots=True)
class AudioEnvironmentError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"cannot prepare audio from {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class AudioCancellationError(Exception):
    path: Path

    def __str__(self) -> str:
        return f"audio normalization cancelled for {self.path}"


@dataclass(slots=True)  # noqa: MUTABLE_OK
class PreparedAudio:
    """Own a normalized temporary WAV until its context or cleanup ends."""

    path: Path
    sample_count: int
    _cleaned: bool = field(default=False, init=False, repr=False)

    @property
    def duration(self) -> float:
        return self.sample_count / SAMPLE_RATE

    def cleanup(self) -> None:
        if self._cleaned:
            return
        _remove_temp(self.path)
        self._cleaned = True

    def __enter__(self) -> PreparedAudio:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.cleanup()
        return False


def normalize_media(input_path: Path) -> PreparedAudio:
    if not input_path.is_file():
        raise AudioEnvironmentError(path=input_path, reason="input is not a file")
    if not os.access(input_path, os.R_OK):
        raise AudioEnvironmentError(path=input_path, reason="input is not readable")
    if shutil.which("ffmpeg") is None:
        raise AudioEnvironmentError(path=input_path, reason="ffmpeg is not available")

    try:
        descriptor, raw_temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(descriptor)
    except OSError as error:
        raise AudioEnvironmentError(
            path=input_path,
            reason=f"cannot allocate temporary WAV: {error}",
        ) from error

    temp_path = Path(raw_temp_path)
    completed = False
    try:
        process = _start_ffmpeg(input_path, temp_path)
        try:
            _, stderr = process.communicate()
        except KeyboardInterrupt as error:
            _stop_process_group(process)
            raise AudioCancellationError(path=input_path) from error
        except BaseException:  # noqa: BROAD_EXCEPT_OK
            _stop_process_group(process)
            raise

        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            reason = "ffmpeg did not produce audio"
            if detail:
                reason = f"{reason}: {detail}"
            raise AudioEnvironmentError(path=input_path, reason=reason)

        sample_count = _validated_sample_count(temp_path, input_path)
        completed = True
        return PreparedAudio(path=temp_path, sample_count=sample_count)
    finally:
        if not completed:
            _remove_temp(temp_path)


def _start_ffmpeg(
    input_path: Path,
    temp_path: Path,
) -> subprocess.Popen[bytes]:
    argv = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(temp_path),
    ]
    try:
        return subprocess.Popen(
            argv,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise AudioEnvironmentError(
            path=input_path,
            reason=f"cannot start ffmpeg: {error}",
        ) from error


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        process.wait()
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
    except ProcessLookupError:
        process.wait()


def _validated_sample_count(temp_path: Path, input_path: Path) -> int:
    try:
        with wave.open(str(temp_path), "rb") as wav_file:
            sample_count = wav_file.getnframes()
            valid_header = (
                wav_file.getframerate() == SAMPLE_RATE
                and wav_file.getnchannels() == CHANNEL_COUNT
                and wav_file.getsampwidth() == SAMPLE_WIDTH
                and wav_file.getcomptype() == "NONE"
            )
    except (EOFError, OSError, wave.Error) as error:
        raise AudioEnvironmentError(
            path=input_path,
            reason=f"invalid normalized WAV: {error}",
        ) from error
    if not valid_header:
        raise AudioEnvironmentError(
            path=input_path,
            reason="normalized WAV must be mono 16 kHz signed 16-bit PCM",
        )
    if sample_count <= 0:
        raise AudioEnvironmentError(
            path=input_path,
            reason="normalized WAV contains no audio samples",
        )
    return sample_count


def _remove_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise AudioEnvironmentError(
            path=path,
            reason=f"cannot remove temporary WAV: {error}",
        ) from error
