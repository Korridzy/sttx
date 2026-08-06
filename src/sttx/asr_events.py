from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class ScanStarted:
    total_samples: int


@dataclass(frozen=True, slots=True)
class ScanAdvanced:
    processed_samples: int
    total_samples: int


@dataclass(frozen=True, slots=True)
class VadSegmentReady:
    index: int
    start_sample: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class DecodeStarted:
    index: int
    start_sample: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class DecodeFinished:
    index: int
    start_sample: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class LanguageReported:
    language: str


@dataclass(frozen=True, slots=True)
class WordCountUpdated:
    word_count: int


@dataclass(frozen=True, slots=True)
class ScanFinished:
    total_samples: int


@dataclass(frozen=True, slots=True)
class TranscriptionSummary:
    total_samples: int
    voiced_samples: int
    vad_segments: int
    decoded_samples: int
    decode_chunks: int
    word_count: int
    language: str
    transcript_segments: int


AsrActivity: TypeAlias = (
    ScanStarted
    | ScanAdvanced
    | VadSegmentReady
    | DecodeStarted
    | DecodeFinished
    | LanguageReported
    | WordCountUpdated
    | ScanFinished
    | TranscriptionSummary
)
ActivityCallback: TypeAlias = Callable[[AsrActivity], None]
