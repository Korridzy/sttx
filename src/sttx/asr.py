from __future__ import annotations

import math
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from sttx.audio import SAMPLE_RATE, PreparedAudio
from sttx.asr_events import (
    ActivityCallback,
    AsrActivity,
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
from sttx.output import Segment, Transcript

VAD_WINDOW_SAMPLES: Final = 512
MAX_CHUNK_SAMPLES: Final = 480_000
CONTROL_TOKENS: Final = frozenset({"", "<blk>", "<blank>", "<s>", "</s>", "<unk>"})
SENTENCE_PUNCTUATION: Final = frozenset({".", "!", "?"})
TIMING_TOLERANCE: Final = 1e-3
# Parakeet TDT reports token timings on the encoder frame grid (subsampling
# factor 8 over a 10 ms feature hop), so the frame covering a chunk's final
# samples may start and end past the chunk itself.
ENCODER_FRAME: Final = 0.08
# A loose sanity bound, not a tight one: legitimate overhang reaches ~0.4 s (one
# frame plus the model's longest duration), and this leaves generous margin so
# it only catches grossly corrupt durations. Do not tighten it towards 0.4
# expecting it to validate durations; that is not what it is for.
MAX_TOKEN_OVERHANG: Final = 1.0

FloatSamples: TypeAlias = NDArray[np.float32]
ProgressCallback: TypeAlias = Callable[[int, int], None]


class RecognitionStream(Protocol):
    @property
    def result(self) -> RecognitionResult: ...

    def accept_waveform(
        self,
        sample_rate: int,
        samples: FloatSamples,
    ) -> None: ...


class RecognitionResult(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def tokens(self) -> Sequence[str]: ...

    @property
    def timestamps(self) -> Sequence[float]: ...

    @property
    def durations(self) -> Sequence[float]: ...

    @property
    def lang(self) -> str: ...


StreamT = TypeVar("StreamT", bound=RecognitionStream)
ResultT = TypeVar("ResultT", bound=RecognitionResult)


class Recognizer(Protocol[StreamT, ResultT]):
    def create_stream(self) -> StreamT: ...

    def decode_stream(self, stream: StreamT) -> None: ...


class VadSegment(Protocol):
    @property
    def start(self) -> int: ...

    @property
    def samples(self) -> FloatSamples: ...


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
    sample_count: int
    activity: ActivityCallback | None
    language: str = ""
    voiced_samples: int = 0
    vad_segments: int = 0
    decoded_samples: int = 0
    decode_chunks: int = 0


def transcribe(
    audio: PreparedAudio,
    *,
    recognizer: Recognizer[StreamT, ResultT],
    vad: VoiceActivityDetector,
    progress: ProgressCallback | None = None,
    activity: ActivityCallback | None = None,
) -> Transcript:
    samples = _read_samples(audio)
    state = _PipelineState(
        words=[],
        sample_count=audio.sample_count,
        activity=activity,
    )
    _emit_activity(activity, ScanStarted(total_samples=audio.sample_count))
    for start in range(0, len(samples), VAD_WINDOW_SAMPLES):
        vad.accept_waveform(samples[start : start + VAD_WINDOW_SAMPLES])
        _drain_vad(vad, recognizer, state)
        processed_samples = min(start + VAD_WINDOW_SAMPLES, audio.sample_count)
        _emit_activity(
            activity,
            ScanAdvanced(
                processed_samples=processed_samples,
                total_samples=audio.sample_count,
            ),
        )
        if progress is not None and processed_samples < audio.sample_count:
            progress(processed_samples, audio.sample_count)
    vad.flush()
    _drain_vad(vad, recognizer, state)
    if progress is not None:
        progress(audio.sample_count, audio.sample_count)
    _emit_activity(activity, ScanFinished(total_samples=audio.sample_count))
    transcript = Transcript(
        language=state.language or "auto",
        duration=audio.duration,
        segments=_sentence_segments(state.words),
    )
    _emit_activity(
        activity,
        TranscriptionSummary(
            total_samples=audio.sample_count,
            voiced_samples=state.voiced_samples,
            vad_segments=state.vad_segments,
            decoded_samples=state.decoded_samples,
            decode_chunks=state.decode_chunks,
            word_count=len(state.words),
            language=transcript.language,
            transcript_segments=len(transcript.segments),
        ),
    )
    return transcript


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
    state: _PipelineState,
) -> None:
    while not vad.empty():
        segment = vad.front
        state.vad_segments += 1
        state.voiced_samples += len(segment.samples)
        _emit_activity(
            state.activity,
            VadSegmentReady(
                index=state.vad_segments,
                start_sample=segment.start,
                sample_count=len(segment.samples),
            ),
        )
        _decode_segment(segment, recognizer, state)
        vad.pop()


def _decode_segment(
    segment: VadSegment,
    recognizer: Recognizer[StreamT, ResultT],
    state: _PipelineState,
) -> None:
    if segment.start < 0 or segment.start + len(segment.samples) > state.sample_count:
        raise TranscriptionError(reason="VAD segment lies outside WAV duration")
    for split_start in range(0, len(segment.samples), MAX_CHUNK_SAMPLES):
        chunk = segment.samples[split_start : split_start + MAX_CHUNK_SAMPLES]
        chunk_start = segment.start + split_start
        state.decode_chunks += 1
        _emit_activity(
            state.activity,
            DecodeStarted(
                index=state.decode_chunks,
                start_sample=chunk_start,
                sample_count=len(chunk),
            ),
        )
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, chunk)
        recognizer.decode_stream(stream)
        state.decoded_samples += len(chunk)
        _emit_activity(
            state.activity,
            DecodeFinished(
                index=state.decode_chunks,
                start_sample=chunk_start,
                sample_count=len(chunk),
            ),
        )
        result = stream.result
        if not state.language and result.lang.strip():
            state.language = result.lang.strip()
            _emit_activity(
                state.activity,
                LanguageReported(language=state.language),
            )
        offset = chunk_start / SAMPLE_RATE
        words = _word_events(result, offset=offset, chunk_duration=len(chunk) / SAMPLE_RATE)
        previous_word_count = len(state.words)
        _append_monotonic(state.words, words, state.sample_count / SAMPLE_RATE)
        if len(state.words) > previous_word_count:
            _emit_activity(
                state.activity,
                WordCountUpdated(word_count=len(state.words)),
            )


def _emit_activity(activity: ActivityCallback | None, event: AsrActivity) -> None:
    if activity is not None:
        activity(event)


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
        raw_ends = tuple(
            start + duration
            for start, duration in zip(timestamps, durations, strict=True)
        )
    else:
        raw_ends = (*timestamps[1:], chunk_duration)
    if any(end > chunk_duration + MAX_TOKEN_OVERHANG for end in raw_ends):
        raise TranscriptionError(reason="token duration runs far past its chunk")
    starts = tuple(min(start, chunk_duration) for start in timestamps)
    ends = tuple(min(end, chunk_duration) for end in raw_ends)

    words: list[WordEvent] = []
    current_text = ""
    current_start = 0.0
    current_end = 0.0
    for token, start, end in zip(tokens, starts, ends, strict=True):
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
            math.isfinite(value) and 0 <= value <= chunk_duration + ENCODER_FRAME
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
