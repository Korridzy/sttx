from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .real_pipeline_artifacts import JsonValue
from .real_pipeline_media import SAMPLE_RATE, read_int16, write_int16
from sttx.asr import MAX_CHUNK_SAMPLES, VoiceActivityDetector, transcribe
from sttx.audio import PreparedAudio
from sttx.cli import _make_vad_from_bundle
from sttx.model import ModelBundle

FloatSamples: TypeAlias = NDArray[np.float32]


class BundleLike(Protocol):
    @property
    def encoder(self) -> Path: ...

    @property
    def decoder(self) -> Path: ...

    @property
    def joiner(self) -> Path: ...

    @property
    def tokens(self) -> Path: ...

    @property
    def silero(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class VadObservation:
    start: int
    sample_count: int

    def to_json(self) -> dict[str, JsonValue]:
        return {"start": self.start, "sample_count": self.sample_count}


@dataclass(frozen=True, slots=True)
class ObservedResult:
    text: str
    tokens: tuple[str, ...]
    timestamps: tuple[float, ...]
    durations: tuple[float, ...] = ()
    lang: str = "en"


@dataclass(frozen=True, slots=True)
class ObservedSegment:
    start: int
    samples: FloatSamples


@dataclass(slots=True)  # noqa: MUTABLE_OK
class ObservedVad:
    segment: ObservedSegment
    drained: bool = False

    def accept_waveform(self, samples: FloatSamples) -> None:
        del samples

    def flush(self) -> None:
        pass

    def empty(self) -> bool:
        return self.drained

    @property
    def front(self) -> ObservedSegment:
        return self.segment

    def pop(self) -> None:
        self.drained = True


@dataclass(slots=True)  # noqa: MUTABLE_OK
class ObservedStream:
    result: ObservedResult
    samples: FloatSamples | None = None

    def accept_waveform(self, sample_rate: int, samples: FloatSamples) -> None:
        if sample_rate != SAMPLE_RATE:
            raise AssertionError(f"unexpected sample rate {sample_rate}")
        self.samples = samples


@dataclass(slots=True)  # noqa: MUTABLE_OK
class ObservingRecognizer:
    chunk_lengths: list[int] = field(default_factory=list)

    def create_stream(self) -> ObservedStream:
        return ObservedStream(
            result=ObservedResult(
                text="Chunk.",
                tokens=(" Chunk", "."),
                timestamps=(0.0, 0.25),
                durations=(0.2, 0.1),
            )
        )

    def decode_stream(self, stream: ObservedStream) -> None:
        if stream.samples is None:
            raise AssertionError("stream was decoded before waveform acceptance")
        self.chunk_lengths.append(len(stream.samples))


def write_vad_positive_continuous_wav(
    bundle: BundleLike,
    source: Path,
    destination: Path,
) -> Path:
    samples = read_int16(source)
    source_segments = observe_vad_segments(bundle, source)
    if not source_segments:
        raise AssertionError("source has no VAD-positive segment")
    first = source_segments[0]
    voiced = samples[first.start : first.start + first.sample_count]
    repetitions = int(np.ceil((SAMPLE_RATE * 36) / max(len(voiced), 1)))
    continuous = np.tile(voiced, repetitions)[: SAMPLE_RATE * 36]
    write_int16(destination, continuous)
    return destination


def observe_vad_segments(
    bundle: BundleLike,
    wav_path: Path,
) -> tuple[VadObservation, ...]:
    model_bundle = ModelBundle(
        encoder=bundle.encoder,
        decoder=bundle.decoder,
        joiner=bundle.joiner,
        tokens=bundle.tokens,
        silero=bundle.silero,
    )
    vad = _make_vad_from_bundle(model_bundle)
    samples = read_int16(wav_path).astype(np.float32) / 32768.0
    observations: list[VadObservation] = []
    for start in range(0, len(samples), 512):
        vad.accept_waveform(samples[start : start + 512])
        observations.extend(_drain(vad))
    vad.flush()
    observations.extend(_drain(vad))
    return tuple(observations)


def observe_hard_split(wav_path: Path) -> dict[str, JsonValue]:
    samples = read_int16(wav_path).astype(np.float32) / 32768.0
    audio = _prepared_audio(wav_path, len(samples))
    recognizer = ObservingRecognizer()
    transcribe(
        audio,
        recognizer=recognizer,
        vad=ObservedVad(ObservedSegment(start=0, samples=samples)),
    )
    return {
        "required": MAX_CHUNK_SAMPLES,
        "chunk_lengths": recognizer.chunk_lengths,
        "first_chunk_exact": recognizer.chunk_lengths[:1] == [MAX_CHUNK_SAMPLES],
        "has_later_chunk": len(recognizer.chunk_lengths) > 1,
    }


def _drain(vad: VoiceActivityDetector) -> list[VadObservation]:
    observations: list[VadObservation] = []
    while not vad.empty():
        segment = vad.front
        observations.append(
            VadObservation(
                start=int(segment.start),
                sample_count=len(segment.samples),
            )
        )
        vad.pop()
    return observations


def _prepared_audio(path: Path, sample_count: int) -> PreparedAudio:
    with wave.open(str(path), "rb") as stream:
        if stream.getframerate() != SAMPLE_RATE:
            raise AssertionError(f"unexpected sample rate {stream.getframerate()}")
    return PreparedAudio(path=path, sample_count=sample_count)
