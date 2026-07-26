from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

JSON_SUFFIX: Final = ".json"
TXT_SUFFIX: Final = ".txt"
TRANSCRIPTIONS_DIR: Final = "transcriptions"
TASK_NAME: Final = "transcribe"


class SegmentJson(TypedDict):
    id: int
    start: float
    end: float
    text: str


class TranscriptJson(TypedDict):
    task: str
    language: str
    duration: float
    text: str
    segments: list[SegmentJson]


@dataclass(frozen=True, slots=True)
class OutputPathError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"invalid output path {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OutputWriteError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"failed to write output {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class Segment:
    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> SegmentJson:
        return {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": normalize_segment_text(self.text),
        }


@dataclass(frozen=True, slots=True)
class Transcript:
    language: str
    duration: float
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        return " ".join(
            text
            for segment in self.segments
            if (text := normalize_segment_text(segment.text))
        )

    def to_dict(self) -> TranscriptJson:
        return {
            "task": TASK_NAME,
            "language": self.language,
            "duration": round(self.duration, 2),
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class OutputPaths:
    json_path: Path
    txt_path: Path


def normalize_segment_text(text: str) -> str:
    return " ".join(text.split())


def output_paths(
    input_path: Path,
    outdir: Path | None,
    name: str | None,
    cwd: Path,
) -> OutputPaths:
    stem = input_path.name
    output_stem = input_path.with_name(stem).stem if name is None else parse_output_name(name)
    output_dir = resolve_output_dir(outdir, cwd)
    ensure_output_dir(output_dir)
    return OutputPaths(
        json_path=output_dir / f"{output_stem}{JSON_SUFFIX}",
        txt_path=output_dir / f"{output_stem}{TXT_SUFFIX}",
    )


def write_outputs(transcript: Transcript, paths: OutputPaths) -> None:
    json_tmp: Path | None = None
    txt_tmp: Path | None = None
    try:
        json_tmp = write_temp(paths.json_path, json_payload(transcript))
        txt_tmp = write_temp(paths.txt_path, transcript.text)
        replace_temp(json_tmp, paths.json_path)
        json_tmp = None
        replace_temp(txt_tmp, paths.txt_path)
        txt_tmp = None
    finally:
        clean_temp(json_tmp)
        clean_temp(txt_tmp)


def parse_output_name(name: str) -> str:
    path = Path(name)
    invalid = (
        not name
        or path.is_absolute()
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or name.endswith(JSON_SUFFIX)
        or name.endswith(TXT_SUFFIX)
    )
    if invalid:
        raise OutputPathError(path=path, reason="name must be one stem without suffix")
    return name


def resolve_output_dir(outdir: Path | None, cwd: Path) -> Path:
    if outdir is None:
        return cwd / TRANSCRIPTIONS_DIR
    if outdir.is_absolute():
        return outdir
    return cwd / outdir


def ensure_output_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OutputPathError(path=path, reason=str(error)) from error
    if not path.is_dir():
        raise OutputPathError(path=path, reason="existing path is not a directory")


def json_payload(transcript: Transcript) -> str:
    return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n"


def write_temp(final_path: Path, contents: str) -> Path:
    prefix = f".{final_path.stem}."
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=final_path.parent,
            prefix=prefix,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(contents)
            return Path(temp_file.name)
    except OSError as error:
        raise OutputWriteError(path=final_path, reason=str(error)) from error


def replace_temp(source: Path, destination: Path) -> None:
    try:
        os.replace(source, destination)
    except OSError as error:
        raise OutputWriteError(path=destination, reason=str(error)) from error


def clean_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise OutputWriteError(path=path, reason=str(error)) from error
