from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, assert_never

import numpy as np

from sttx.asr import FloatSamples, RecognitionResult, RecognitionStream, Recognizer, VadSegment, VoiceActivityDetector
from tests.integration.timing_lattice import ProductionRun, RawObservation, quantize_us
from tests.integration.qualification_validation import require

StreamT = TypeVar("StreamT", bound=RecognitionStream)
ResultT = TypeVar("ResultT", bound=RecognitionResult)
EventKind = Literal["feed", "empty", "front", "pop", "flush", "create", "accept", "decode", "result"]


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    value: int


@dataclass(frozen=True, slots=True)
class Front:
    start: int
    samples: FloatSamples


def raw_observation(result: RecognitionResult) -> RawObservation:
    return {"tokens": list(result.tokens), "timestamps_us": [quantize_us(value) for value in result.timestamps],
            "durations_us": [quantize_us(value) for value in result.durations], "text": result.text, "lang": result.lang}


@dataclass(slots=True)
class Recording:
    """Mutable append-only trace; validation never calls the wrapped backends."""

    probe: str
    sample_count: int
    events: list[Event] = field(default_factory=list)
    fronts: list[Front] = field(default_factory=list)
    streams: list[RecognitionStream] = field(default_factory=list)
    accepted: list[FloatSamples] = field(default_factory=list)
    results: list[RawObservation] = field(default_factory=list)

    def wrap(self, recognizer: Recognizer[StreamT, ResultT], vad: VoiceActivityDetector
             ) -> tuple[RecordingRecognizer[StreamT, ResultT], RecordingVad]:
        return RecordingRecognizer(recognizer, self), RecordingVad(vad, self)

    def validate(self) -> list[ProductionRun]:
        require(bool(self.events) and self.events[-1] == Event("empty", 1), "missing final drain")
        require(len({id(stream) for stream in self.streams}) == len(self.streams), "streams are not fresh")
        runs: list[ProductionRun] = []
        fed = flushes = segment_index = chunk_index = offset = stream_index = 0
        previous_end = 0
        drained = True
        front: Front | None = None
        stage = "idle"
        previous = Event("empty", 1)
        for event in self.events:
            match event.kind:
                case "feed":
                    require(drained and flushes == 0 and stage == "idle", "feed before drain or after flush")
                    require(0 < event.value <= 512, "invalid feed size")
                    fed += event.value
                    drained = False
                case "empty":
                    require(front is None and stage == "idle" and event.value in (0, 1), "empty during active front")
                    drained = event.value == 1
                case "flush":
                    require(drained and front is None and flushes == 0 and fed == self.sample_count, "invalid flush order")
                    flushes += 1
                    drained = False
                case "front":
                    require(previous == Event("empty", 0) and front is None, "front without nonempty queue")
                    require(event.value == segment_index and event.value < len(self.fronts), "front identity")
                    front = self.fronts[event.value]
                    require(0 <= front.start < front.start + len(front.samples) <= self.sample_count, "front bounds")
                    require(front.start >= previous_end, "nonmonotonic front bounds")
                    offset = chunk_index = 0
                case "create":
                    require(front is not None and stage == "idle" and event.value == stream_index, "create order")
                    stage = "created"
                case "accept":
                    require(front is not None and stage == "created" and event.value == stream_index, "accept order")
                    require(stream_index < len(self.accepted), "missing accepted chunk")
                    samples = self.accepted[stream_index]
                    require(0 < len(samples) <= 480000, "chunk size")
                    assert front is not None
                    require(np.array_equal(samples, front.samples[offset:offset + len(samples)]), "chunk data mismatch")
                    require(len(samples) == min(480000, len(front.samples) - offset), "incorrect chunk split")
                    stage = "accepted"
                case "decode":
                    require(stage == "accepted" and event.value == stream_index, "decode order")
                    stage = "decoded"
                case "result":
                    require(front is not None and stage == "decoded" and event.value == stream_index, "result order")
                    require(stream_index < len(self.results), "missing raw result")
                    assert front is not None
                    count = len(self.accepted[stream_index])
                    runs.append({"probe": self.probe, "segment_index": segment_index, "chunk_index": chunk_index,
                        "start_sample": front.start + offset, "sample_count": count, **self.results[stream_index]})
                    offset += count
                    chunk_index += 1
                    stream_index += 1
                    stage = "idle"
                case "pop":
                    require(front is not None and stage == "idle" and chunk_index > 0, "pop without decoded front")
                    assert front is not None
                    require(offset == len(front.samples) and event.value == segment_index, "pop before complete decode")
                    previous_end = front.start + offset
                    segment_index += 1
                    front = None
                case unreachable:
                    assert_never(unreachable)
            previous = event
        require(flushes == 1 and fed == self.sample_count and drained and front is None, "incomplete scan")
        require(segment_index == len(self.fronts) and stream_index == len(self.streams)
                == len(self.accepted) == len(self.results), "incomplete recording")
        return runs


@dataclass(frozen=True, slots=True)
class RecordingStream(Generic[StreamT]):
    raw: StreamT
    recording: Recording
    index: int

    def accept_waveform(self, sample_rate: int, samples: FloatSamples) -> None:
        require(sample_rate == 16000, "incorrect sample rate")
        self.recording.events.append(Event("accept", self.index))
        self.recording.accepted.append(samples.copy())
        self.raw.accept_waveform(sample_rate, samples)

    @property
    def result(self) -> RecognitionResult:
        result = self.raw.result
        self.recording.events.append(Event("result", self.index))
        self.recording.results.append(raw_observation(result))
        return result


@dataclass(frozen=True, slots=True)
class RecordingRecognizer(Generic[StreamT, ResultT]):
    raw: Recognizer[StreamT, ResultT]
    recording: Recording

    def create_stream(self) -> RecordingStream[StreamT]:
        raw = self.raw.create_stream()
        index = len(self.recording.streams)
        self.recording.streams.append(raw)
        self.recording.events.append(Event("create", index))
        return RecordingStream(raw, self.recording, index)

    def decode_stream(self, stream: RecordingStream[StreamT]) -> None:
        self.recording.events.append(Event("decode", stream.index))
        self.raw.decode_stream(stream.raw)


@dataclass(frozen=True, slots=True)
class RecordingVad:
    raw: VoiceActivityDetector
    recording: Recording

    def accept_waveform(self, samples: FloatSamples) -> None:
        self.recording.events.append(Event("feed", len(samples)))
        self.raw.accept_waveform(samples)

    def empty(self) -> bool:
        result = self.raw.empty()
        self.recording.events.append(Event("empty", int(result)))
        return result

    def flush(self) -> None:
        self.recording.events.append(Event("flush", 0))
        self.raw.flush()

    @property
    def front(self) -> VadSegment:
        segment = self.raw.front
        self.recording.events.append(Event("front", len(self.recording.fronts)))
        self.recording.fronts.append(Front(segment.start, segment.samples.copy()))
        return segment

    def pop(self) -> None:
        self.recording.events.append(Event("pop", len(self.recording.fronts) - 1))
        self.raw.pop()
