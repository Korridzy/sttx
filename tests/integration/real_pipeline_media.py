from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

SAMPLE_RATE: Final = 16_000
FFMPEG_TIMEOUT: Final = 60.0


@dataclass(frozen=True, slots=True)
class TranscriptCheck:
    text: str
    duration: float
    segment_count: int
    monotonic: bool
    json_txt_equal: bool


def ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AssertionError("ffmpeg is required for real pipeline QA")
    return ffmpeg


def run_ffmpeg(command: Sequence[str]) -> None:
    completed = subprocess.run(
        [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", *command],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=FFMPEG_TIMEOUT,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.decode("utf-8", "replace"))


def convert_media(source: Path, destination: Path) -> Path:
    match destination.suffix:
        case ".wav":
            run_ffmpeg(["-i", str(source), "-ac", "1", "-ar", "16000", "-y", str(destination)])
        case ".mp3":
            run_ffmpeg(["-i", str(source), "-codec:a", "libmp3lame", "-y", str(destination)])
        case ".ogg":
            run_ffmpeg(["-i", str(source), "-codec:a", "libvorbis", "-y", str(destination)])
        case ".mp4":
            run_ffmpeg(
                [
                    "-i",
                    str(source),
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=16x16:d=1",
                    "-shortest",
                    "-c:a",
                    "aac",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(destination),
                ]
            )
        case unmatched:
            raise AssertionError(f"unsupported media suffix {unmatched}")
    return destination


def write_two_utterance_wav(source: Path, destination: Path) -> Path:
    samples = read_int16(source)
    silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
    joined = np.concatenate([samples, silence, samples])
    write_int16(destination, joined)
    return destination


def write_continuous_wav(source: Path, destination: Path) -> Path:
    samples = read_int16(source)
    repetitions = int(np.ceil((SAMPLE_RATE * 32) / max(len(samples), 1)))
    continuous = np.tile(samples, repetitions)[: SAMPLE_RATE * 32]
    write_int16(destination, continuous)
    return destination


def write_silence_wav(destination: Path) -> Path:
    write_int16(destination, np.zeros(SAMPLE_RATE, dtype=np.int16))
    return destination


def write_no_audio_mp4(destination: Path) -> Path:
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(destination),
        ]
    )
    return destination


def read_transcript(json_path: Path, txt_path: Path) -> TranscriptCheck:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    segments = payload["segments"]
    starts = [float(segment["start"]) for segment in segments]
    ends = [float(segment["end"]) for segment in segments]
    monotonic = all(left <= right for left, right in zip(starts, starts[1:]))
    monotonic = monotonic and all(start <= end for start, end in zip(starts, ends))
    text = str(payload["text"])
    return TranscriptCheck(
        text=text,
        duration=float(payload["duration"]),
        segment_count=len(segments),
        monotonic=monotonic,
        json_txt_equal=txt_path.read_text(encoding="utf-8") == text,
    )


def read_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if stream.getframerate() != SAMPLE_RATE or stream.getnchannels() != 1:
            return _read_normalized_int16(path)
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)


def write_int16(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(samples.astype(np.int16).tobytes())


def _read_normalized_int16(path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as normalized:
        run_ffmpeg(["-i", str(path), "-ac", "1", "-ar", "16000", "-y", normalized.name])
        with wave.open(normalized.name, "rb") as stream:
            return np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
