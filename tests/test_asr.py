from __future__ import annotations

import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sttx.asr import TranscriptionError, transcribe
from sttx.audio import PreparedAudio
from sttx.output import TranscriptJson

SAMPLE_RATE = 16_000
MAX_CHUNK_SAMPLES = 480_000
type FloatSamples = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FakeResult:
    text: str
    tokens: tuple[str, ...]
    timestamps: tuple[float, ...]
    durations: tuple[float, ...] = ()
    lang: str = ""


@dataclass(frozen=True, slots=True)
class FakeVadSegment:
    start: int
    samples: FloatSamples


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeVad:
    pending: list[FakeVadSegment]
    fed_sizes: list[int] = field(default_factory=list)
    flushed: bool = False
    ready: list[FakeVadSegment] = field(default_factory=list)

    def accept_waveform(self, samples: FloatSamples) -> None:
        self.fed_sizes.append(len(samples))

    def flush(self) -> None:
        self.flushed = True
        self.ready.extend(self.pending)

    def is_empty(self) -> bool:
        return not self.ready

    @property
    def front(self) -> FakeVadSegment:
        return self.ready[0]

    def pop(self) -> None:
        self.ready.pop(0)


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeStream:
    samples: FloatSamples | None = None

    def accept_waveform(self, sample_rate: int, samples: FloatSamples) -> None:
        assert sample_rate == SAMPLE_RATE
        self.samples = samples


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeRecognizer:
    results: list[FakeResult]
    chunk_lengths: list[int] = field(default_factory=list)

    def create_stream(self) -> FakeStream:
        return FakeStream()

    def decode_stream(self, stream: FakeStream) -> None:
        assert stream.samples is not None
        self.chunk_lengths.append(len(stream.samples))

    def get_result(self, stream: FakeStream) -> FakeResult:
        assert stream.samples is not None
        return self.results.pop(0)


def _prepared_wav(tmp_path: Path, sample_count: int) -> PreparedAudio:
    path = tmp_path / "prepared.wav"
    samples = np.zeros(sample_count, dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())
    return PreparedAudio(path=path, sample_count=sample_count)


def _segment(start: int, sample_count: int) -> FakeVadSegment:
    return FakeVadSegment(
        start=start,
        samples=np.zeros(sample_count, dtype=np.float32),
    )


def _run(
    tmp_path: Path,
    *,
    sample_count: int,
    segments: Sequence[FakeVadSegment],
    results: Sequence[FakeResult],
) -> tuple[FakeVad, FakeRecognizer, TranscriptJson]:
    vad = FakeVad(pending=list(segments))
    recognizer = FakeRecognizer(results=list(results))
    transcript = transcribe(
        _prepared_wav(tmp_path, sample_count),
        recognizer=recognizer,
        vad=vad,
    )
    return vad, recognizer, transcript.to_dict()


def test_sentencepiece_tokens_become_timed_words(tmp_path: Path) -> None:
    result = FakeResult(
        text="Hello world.",
        tokens=("▁Hel", "lo", "▁world", "."),
        timestamps=(0.0, 0.1, 0.5, 0.9),
        durations=(0.1, 0.4, 0.4, 0.1),
    )

    _, _, payload = _run(
        tmp_path,
        sample_count=16_000,
        segments=(_segment(0, 16_000),),
        results=(result,),
    )

    assert payload["segments"] == [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello world."}
    ]


def test_sentence_boundaries_and_trailing_words(tmp_path: Path) -> None:
    result = FakeResult(
        text="One. Two trailing",
        tokens=(" One", ".", " Two", " trailing"),
        timestamps=(0.0, 0.2, 0.5, 0.8),
    )

    _, _, payload = _run(
        tmp_path,
        sample_count=16_000,
        segments=(_segment(0, 16_000),),
        results=(result,),
    )

    assert payload["segments"] == [
        {"id": 0, "start": 0.0, "end": 0.5, "text": "One."},
        {"id": 1, "start": 0.5, "end": 1.0, "text": "Two trailing"},
    ]


def test_vad_flush_emits_final_speech(tmp_path: Path) -> None:
    vad, recognizer, payload = _run(
        tmp_path,
        sample_count=800,
        segments=(_segment(0, 800),),
        results=(
            FakeResult(
                text="Final",
                tokens=(" Final",),
                timestamps=(0.0,),
            ),
        ),
    )

    assert vad.flushed is True
    assert recognizer.chunk_lengths == [800]
    assert payload["text"] == "Final"


def test_vad_windows_never_exceed_30_seconds(tmp_path: Path) -> None:
    vad, recognizer, _ = _run(
        tmp_path,
        sample_count=MAX_CHUNK_SAMPLES + 17,
        segments=(_segment(0, MAX_CHUNK_SAMPLES + 17),),
        results=(
            FakeResult("First", (" First",), (0.0,)),
            FakeResult("Second", (" Second",), (0.0,)),
        ),
    )

    assert max(vad.fed_sizes) <= 512
    assert recognizer.chunk_lengths == [MAX_CHUNK_SAMPLES, 17]


def test_hard_split_offsets_include_subchunk_start(tmp_path: Path) -> None:
    vad_start = 8_000
    _, _, payload = _run(
        tmp_path,
        sample_count=vad_start + MAX_CHUNK_SAMPLES + 16_000,
        segments=(_segment(vad_start, MAX_CHUNK_SAMPLES + 16_000),),
        results=(
            FakeResult("First.", (" First", "."), (0.0, 29.0)),
            FakeResult("Second.", (" Second", "."), (0.0, 0.5)),
        ),
    )

    assert payload["segments"][1]["start"] == 30.5


def test_global_offsets_are_monotonic(tmp_path: Path) -> None:
    _, _, payload = _run(
        tmp_path,
        sample_count=48_000,
        segments=(_segment(0, 16_000), _segment(32_000, 16_000)),
        results=(
            FakeResult("Early.", (" Early", "."), (0.0, 0.8)),
            FakeResult("Late.", (" Late", "."), (0.0, 0.8)),
        ),
    )

    segments = payload["segments"]
    assert segments[0]["end"] <= segments[1]["start"]
    assert segments[1]["end"] <= payload["duration"]


def test_silence_is_empty_success(tmp_path: Path) -> None:
    _, recognizer, payload = _run(
        tmp_path,
        sample_count=16_000,
        segments=(),
        results=(),
    )

    assert recognizer.chunk_lengths == []
    assert payload == {
        "task": "transcribe",
        "language": "auto",
        "duration": 1.0,
        "text": "",
        "segments": [],
    }


def test_language_falls_back_to_auto(tmp_path: Path) -> None:
    _, _, payload = _run(
        tmp_path,
        sample_count=16_000,
        segments=(_segment(0, 8_000),),
        results=(FakeResult("Text", (" Text",), (0.0,), lang=""),),
    )

    assert payload["language"] == "auto"


def test_first_reported_language_is_used(tmp_path: Path) -> None:
    _, _, payload = _run(
        tmp_path,
        sample_count=16_000,
        segments=(_segment(0, 8_000),),
        results=(FakeResult("Text", (" Text",), (0.0,), lang=" en "),),
    )

    assert payload["language"] == "en"


def test_control_only_result_is_empty(tmp_path: Path) -> None:
    _, _, payload = _run(
        tmp_path,
        sample_count=16_000,
        segments=(_segment(0, 8_000),),
        results=(
            FakeResult("", ("", "<blk>", "</s>"), (0.0, 0.1, 0.2)),
        ),
    )

    assert payload["segments"] == []


@pytest.mark.parametrize(
    "result",
    [
        FakeResult("Bad", (" Bad",), (-0.1,)),
        FakeResult("Bad timing", (" Bad", " timing"), (0.4, 0.2)),
        FakeResult("Bad", (" Bad",), (0.0,), durations=(-0.1,)),
        FakeResult("Bad", (" Bad",), (0.9,), durations=(0.2,)),
        FakeResult("Mismatch", (" Mis", "match"), (0.0,)),
        FakeResult("Mismatch", (" Mismatch",), (0.0,), durations=(0.1, 0.2)),
        FakeResult(".", (".",), (0.0,)),
    ],
)
def test_invalid_timing_is_transcription_error(
    tmp_path: Path,
    result: FakeResult,
) -> None:
    with pytest.raises(TranscriptionError):
        _run(
            tmp_path,
            sample_count=16_000,
            segments=(_segment(0, 16_000),),
            results=(result,),
        )
