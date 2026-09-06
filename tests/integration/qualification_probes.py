from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from huggingface_hub import hf_hub_download

from sttx import asr, backend_contract as contract, cli
from sttx.audio import PreparedAudio, normalize_media
from sttx.model import ModelBundle, resolve_bundle
from sttx.output import Transcript
from tests.integration import qualification_schema as schema
from tests.integration import timing_lattice as lattice
from tests.integration.qualification_recording import Recording, raw_observation
from tests.integration.qualification_validation import max_overhang, require

StreamT = TypeVar("StreamT", bound=asr.RecognitionStream)
ResultT = TypeVar("ResultT", bound=asr.RecognitionResult)


@dataclass(frozen=True, slots=True)
class Backends:
    recognizer: asr.Recognizer[asr.RecognitionStream, asr.RecognitionResult]
    vad: asr.VoiceActivityDetector


def make_backends(bundle: ModelBundle) -> Backends:
    return Backends(cli._make_recognizer_from_bundle(bundle), cli._make_vad_from_bundle(bundle))


def collect_raw(audio: PreparedAudio, recognizer: asr.Recognizer[StreamT, ResultT]) -> list[lattice.RunObservation]:
    samples = asr._read_samples(audio)
    streams: list[StreamT] = []
    runs: list[lattice.RunObservation] = []
    for prefix in lattice.VARIANTS:
        stream = recognizer.create_stream()
        require(all(stream is not previous for previous in streams), "raw streams are not fresh")
        streams.append(stream)
        stream.accept_waveform(16000, np.concatenate((np.zeros(prefix, dtype=np.float32), samples)))
        recognizer.decode_stream(stream)
        runs.append({"variant": prefix, **raw_observation(stream.result)})
    return runs


def transcript_observation(probe: str, transcript: Transcript) -> lattice.TranscriptObservation:
    return {"probe": probe, "language": transcript.language, "duration_us": lattice.quantize_us(transcript.duration),
        "text": transcript.text, "segments": [{"id": segment.id, "start_us": lattice.quantize_us(segment.start),
            "end_us": lattice.quantize_us(segment.end), "text": segment.text} for segment in transcript.segments]}


def collect_production(audio: PreparedAudio, bundle: ModelBundle, probe: str
                       ) -> tuple[Recording, list[lattice.ProductionRun], lattice.TranscriptObservation]:
    backends = make_backends(bundle)
    recording = Recording(probe, audio.sample_count)
    recognizer, vad = recording.wrap(backends.recognizer, backends.vad)
    transcript = asr.transcribe(audio, recognizer=recognizer, vad=vad)
    runs = recording.validate()
    return recording, runs, transcript_observation(probe, transcript)


def write_long_probe(original: PreparedAudio, recording: Recording, destination: PreparedAudio) -> None:
    require(bool(recording.fronts), "missing fixture speech")
    front = recording.fronts[0]
    with wave.open(str(original.path), "rb") as source:
        source.setpos(front.start)
        pcm = source.readframes(len(front.samples))
    require(len(pcm) == len(front.samples) * 2 and bool(pcm), "invalid fixture speech slice")
    tiled = (pcm * ((576000 * 2 + len(pcm) - 1) // len(pcm)))[:576000 * 2]
    with wave.open(str(destination.path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(tiled)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def collect_observations(scratch: Path, expected: schema.Expectations) -> Iterator[schema.Observations]:
    bundle = resolve_bundle()
    fixture_path = Path(hf_hub_download(repo_id=contract.FIXTURE_REPO_ID, revision=contract.FIXTURE_REVISION,
        filename=contract.FIXTURE_FILENAME, cache_dir=scratch / "fixture-cache",
        local_files_only=False, force_download=True))
    fixture: schema.FixtureIdentity = {"repo": contract.FIXTURE_REPO_ID, "revision": contract.FIXTURE_REVISION,
        "name": contract.FIXTURE_FILENAME, "sha256": file_sha256(fixture_path)}
    model: schema.ModelIdentity = {"repo": contract.PARAKEET_REPO_ID, "revision": bundle.encoder.parent.name,
        "sha256": {path.name: file_sha256(path) for path in (
            bundle.encoder, bundle.decoder, bundle.joiner, bundle.tokens, bundle.silero)}}
    with normalize_media(fixture_path) as original, PreparedAudio(scratch / "long.wav", 576000) as long_audio:
        runs = collect_raw(original, cli._make_recognizer_from_bundle(bundle))
        recording, original_runs, original_transcript = collect_production(original, bundle, "original")
        write_long_probe(original, recording, long_audio)
        _, long_runs, long_transcript = collect_production(long_audio, bundle, "long")
        payload: lattice.SignedPayload = {"runs": runs, "production_runs": original_runs + long_runs,
                                          "production_transcripts": [original_transcript, long_transcript]}
        reason: str | None = None
        quantum: int | None = None
        try:
            quantum = lattice.derive_quantum_us([time for run in runs for time in run["timestamps_us"]])
        except lattice.ObservationError as error:
            reason = str(error)
        yield {"record_kind": "qualification_candidate", "schema_version": 1,
            "fingerprint": expected.fingerprint, "probe_version": contract.PROBE_VERSION,
            "model": model, "fixture": fixture,
            "dependencies": {name: importlib.metadata.version(name) for name in expected.dependencies},
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "derived_quantum_us": quantum, "inconclusive_reason": reason,
            "observed_max_overhang_us": max_overhang(payload),
            "signature_sha256": lattice.signature_sha256(payload), **payload}
