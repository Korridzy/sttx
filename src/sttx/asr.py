from __future__ import annotations

import math
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from sttx.audio import SAMPLE_RATE, PreparedAudio
from sttx.output import Segment, Transcript

VAD_WINDOW_SAMPLES: Final = 512
MAX_CHUNK_SAMPLES: Final = 480_000
CONTROL_TOKENS: Final = frozenset({"", "<blk>", "<blank>", "<s>", "</s>", "<unk>"})
SENTENCE_PUNCTUATION: Final = frozenset({".", "!", "?"})
TIMING_TOLERANCE: Final = 1e-3

FloatSamples: TypeAlias = NDArray[np.float32]
ProgressCallback: TypeAlias = Callable[[int, int], None]


class RecognitionStream(Protocol):
    result: RecognitionResult

    def accept_waveform(
        self,
        sample_rate: int,
        samples: FloatSamples,
    ) -> None: ...


class RecognitionResult(Protocol):
    text: str
    tokens: Sequence[str]
    timestamps: Sequence[float]
    durations: Sequence[float]
    lang: str


StreamT = TypeVar("StreamT", bound=RecognitionStream)
ResultT = TypeVar("ResultT", bound=RecognitionResult)


class Recognizer(Protocol[StreamT, ResultT]):
    def create_stream(self) -> StreamT: ...

    def decode_stream(self, stream: StreamT) -> None: ...


class VadSegment(Protocol):
    start: int
    samples: FloatSamples


class VoiceActivityDetector(Protocol):
    def accept_waveform(self, samples: FloatSamples) -> None: ...

    def flush(self) -> None: ...

    def empty(self) -> bool: ...

    @property
    def front(self) -> VadSegment: ...

    def pop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TranscriptionError(Exception):
    reason: str

    def __str__(self) -> str:
        return f"transcription failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class WordEvent:
    text: str
    start: float
    end: float


@dataclass(slots=True)  # noqa: MUTABLE_OK
class _PipelineState:
    words: list[WordEvent]
    language: str = ""


def transcribe(
    audio: PreparedAudio,
    *,
    recognizer: Recognizer[StreamT, ResultT],
    vad: VoiceActivityDetector,
    progress: ProgressCallback | None = None,
) -> Transcript:
    samples = _read_samples(audio)
    state = _PipelineState(words=[])
    for start in range(0, len(samples), VAD_WINDOW_SAMPLES):
        vad.accept_waveform(samples[start : start + VAD_WINDOW_SAMPLES])
        _drain_vad(vad, recognizer, audio.sample_count, state)
        processed_samples = min(start + VAD_WINDOW_SAMPLES, audio.sample_count)
        if progress is not None and processed_samples < audio.sample_count:
            progress(processed_samples, audio.sample_count)
    vad.flush()
    _drain_vad(vad, recognizer, audio.sample_count, state)
    if progress is not None:
        progress(audio.sample_count, audio.sample_count)
    return Transcript(
        language=state.language or "auto",
        duration=audio.duration,
        segments=_sentence_segments(state.words),
    )


def _read_samples(audio: PreparedAudio) -> FloatSamples:
    try:
        with wave.open(str(audio.path), "rb") as wav_file:
            raw_samples = wav_file.readframes(wav_file.getnframes())
    except (EOFError, OSError, wave.Error) as error:
        raise TranscriptionError(reason=f"cannot read normalized WAV: {error}") from error
    samples = np.frombuffer(raw_samples, dtype=np.int16)
    if len(samples) != audio.sample_count:
        raise TranscriptionError(
            reason=(
                "normalized WAV sample count changed: "
                f"{len(samples)} != {audio.sample_count}"
            )
        )
    return samples.astype(np.float32) / 32768.0


def _drain_vad(
    vad: VoiceActivityDetector,
    recognizer: Recognizer[StreamT, ResultT],
    wav_sample_count: int,
    state: _PipelineState,
) -> None:
    while not vad.empty():
        segment = vad.front
        _decode_segment(segment, recognizer, wav_sample_count, state)
        vad.pop()


def _decode_segment(
    segment: VadSegment,
    recognizer: Recognizer[StreamT, ResultT],
    wav_sample_count: int,
    state: _PipelineState,
) -> None:
    if segment.start < 0 or segment.start + len(segment.samples) > wav_sample_count:
        raise TranscriptionError(reason="VAD segment lies outside WAV duration")
    for split_start in range(0, len(segment.samples), MAX_CHUNK_SAMPLES):
        chunk = segment.samples[split_start : split_start + MAX_CHUNK_SAMPLES]
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, chunk)
        recognizer.decode_stream(stream)
        result = stream.result
        if not state.language and result.lang.strip():
            state.language = result.lang.strip()
        offset = (segment.start + split_start) / SAMPLE_RATE
        words = _word_events(result, offset=offset, chunk_duration=len(chunk) / SAMPLE_RATE)
        _append_monotonic(state.words, words, wav_sample_count / SAMPLE_RATE)


def _word_events(
    result: RecognitionResult,
    *,
    offset: float,
    chunk_duration: float,
) -> tuple[WordEvent, ...]:
    tokens = tuple(result.tokens)
    timestamps = tuple(float(timestamp) for timestamp in result.timestamps)
    durations = tuple(float(duration) for duration in result.durations)
    if len(tokens) != len(timestamps):
        raise TranscriptionError(reason="token/timestamp cardinality mismatch")
    if durations and len(durations) != len(tokens):
        raise TranscriptionError(reason="token/duration cardinality mismatch")
    if not tokens:
        return ()
    if not _valid_starts(timestamps, chunk_duration):
        raise TranscriptionError(reason="invalid token timestamps")
    if durations:
        if not all(math.isfinite(value) and value >= 0 for value in durations):
            raise TranscriptionError(reason="invalid token durations")
        ends = tuple(
            start + duration
            for start, duration in zip(timestamps, durations, strict=True)
        )
    else:
        ends = (*timestamps[1:], chunk_duration)
    if any(
        end < start
        or end > chunk_duration + TIMING_TOLERANCE
        or not math.isfinite(end)
        for start, end in zip(timestamps, ends, strict=True)
    ):
        raise TranscriptionError(reason="token end lies outside its chunk")

    words: list[WordEvent] = []
    current_text = ""
    current_start = 0.0
    current_end = 0.0
    for token, start, end in zip(tokens, timestamps, ends, strict=True):
        if token in CONTROL_TOKENS:
            continue
        boundary = token.startswith("▁") or token.startswith(" ")
        if boundary:
            piece = token[1:] if token.startswith("▁") else token.lstrip(" ")
            if current_text:
                words.append(
                    WordEvent(current_text, offset + current_start, offset + current_end)
                )
            if not piece:
                current_text = ""
                continue
            current_text = piece
            current_start = start
            current_end = end
            continue
        if not current_text:
            if token in SENTENCE_PUNCTUATION:
                words.append(WordEvent(token, offset + start, offset + end))
            else:
                current_text, current_start, current_end = token, start, end
            continue
        current_text += token
        current_end = end
        if token in SENTENCE_PUNCTUATION:
            words.append(
                WordEvent(current_text, offset + current_start, offset + current_end)
            )
            current_text = ""
    if current_text:
        words.append(WordEvent(current_text, offset + current_start, offset + current_end))
    return tuple(words)


def _valid_starts(timestamps: tuple[float, ...], chunk_duration: float) -> bool:
    return (
        all(
            math.isfinite(value) and 0 <= value <= chunk_duration + TIMING_TOLERANCE
            for value in timestamps
        )
        and all(left <= right for left, right in zip(timestamps, timestamps[1:]))
    )


def _append_monotonic(
    accumulated: list[WordEvent],
    incoming: tuple[WordEvent, ...],
    wav_duration: float,
) -> None:
    previous_end = accumulated[-1].end if accumulated else 0.0
    for word in incoming:
        if (
            word.start + TIMING_TOLERANCE < previous_end
            or word.end < word.start
            or word.end > wav_duration + TIMING_TOLERANCE
        ):
            raise TranscriptionError(reason="global word timing is not monotonic")
        if word.text in SENTENCE_PUNCTUATION and accumulated:
            previous = accumulated[-1]
            accumulated[-1] = WordEvent(
                text=f"{previous.text}{word.text}",
                start=previous.start,
                end=word.end,
            )
            previous_end = word.end
            continue
        accumulated.append(word)
        previous_end = word.end


def _sentence_segments(words: list[WordEvent]) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    sentence: list[WordEvent] = []
    for word in words:
        sentence.append(word)
        if word.text[-1:] in SENTENCE_PUNCTUATION:
            segments.append(_segment(len(segments), sentence))
            sentence = []
    if sentence:
        segments.append(_segment(len(segments), sentence))
    return tuple(segments)


def _segment(segment_id: int, words: list[WordEvent]) -> Segment:
    return Segment(
        id=segment_id,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words),
    )
