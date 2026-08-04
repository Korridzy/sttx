# noqa: SIZE_OK — one indivisible external compatibility evidence probe
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import shutil
import subprocess
import sys
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeGuard, TypedDict

import numpy as np
import pytest
import sherpa_onnx
from huggingface_hub import hf_hub_download

from sttx.audio import PreparedAudio, normalize_media
from sttx.model import PARAKEET_REPO_ID, SILERO_URL, ModelBundle, resolve_bundle

SAMPLE_RATE = 16_000
CONTROL_TOKENS = frozenset({"", "<blk>", "<blank>", "<s>", "</s>", "<unk>"})
PUNCTUATION = frozenset({".", "!", "?"})
RUNTIME_PACKAGES = (
    "huggingface-hub",
    "numpy",
    "pytest",
    "sherpa-onnx",
    "sherpa-onnx-bin",
    "sttx",
)

type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | Sequence[JsonValue]
    | Mapping[str, JsonValue]
)


class IdentityWriter(Protocol):
    def __call__(self, payload: Mapping[str, JsonValue]) -> None: ...


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


class ResultCandidate(Protocol):
    pass


class VadSegmentEvidence(TypedDict):
    start: int
    start_seconds: float
    sample_count: int
    chunk_end_seconds: float


@dataclass(frozen=True, slots=True)
class ContractObservation:
    reconstructed_text: str
    control_tokens: tuple[str, ...]
    token_end_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FakeResult:
    text: str
    tokens: tuple[str, ...]
    timestamps: tuple[float, ...]
    durations: tuple[float, ...] = ()
    lang: str = ""


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _assert_result_contract(
    candidate: ResultCandidate,
    *,
    final_chunk_end_seconds: float,
) -> ContractObservation:
    result = _require_recognition_result(candidate)
    required = ("text", "tokens", "timestamps", "durations", "lang")
    missing = [name for name in required if not hasattr(candidate, name)]
    assert not missing, f"binding result is missing required fields: {missing}"

    text = result.text
    tokens = tuple(result.tokens)
    timestamps = tuple(float(value) for value in result.timestamps)
    durations = tuple(float(value) for value in result.durations)
    assert text.strip(), "binding result text is empty"
    assert tokens, "binding result tokens are empty"
    assert len(tokens) == len(timestamps), (
        "token/timestamp cardinality mismatch: "
        f"{len(tokens)} tokens != {len(timestamps)} timestamps"
    )
    assert all(math.isfinite(value) and value >= 0 for value in timestamps), (
        f"timestamps must be finite nonnegative seconds: {timestamps}"
    )
    assert list(timestamps) == sorted(timestamps), (
        f"timestamps are not monotonic seconds: {timestamps}"
    )
    assert not durations or len(durations) == len(tokens), (
        "token/duration cardinality mismatch: "
        f"{len(tokens)} tokens != {len(durations)} durations"
    )

    words: list[str] = []
    controls: list[str] = []
    for token in tokens:
        if token in CONTROL_TOKENS:
            controls.append(token)
        elif token.startswith(("▁", " ")):
            piece = token.removeprefix("▁").lstrip(" ")
            words.append(piece)
        elif words:
            words[-1] += token
        else:
            words.append(token)

    reconstructed = " ".join(words)
    assert _normalize(reconstructed) == _normalize(text), (
        "token grammar reconstruction differs from recognizer text: "
        f"{reconstructed!r} != {text!r}"
    )

    if durations:
        assert all(math.isfinite(value) and value >= 0 for value in durations), (
            f"durations must be finite nonnegative seconds: {durations}"
        )
        ends = tuple(start + duration for start, duration in zip(timestamps, durations))
    else:
        ends = (*timestamps[1:], final_chunk_end_seconds)
    assert all(end >= start for start, end in zip(timestamps, ends)), (
        f"token ends precede token starts: {tuple(zip(timestamps, ends))}"
    )
    assert ends[-1] <= final_chunk_end_seconds + 1e-3, (
        "final token end exceeds the final VAD chunk end: "
        f"{ends[-1]} > {final_chunk_end_seconds}"
    )
    return ContractObservation(reconstructed, tuple(controls), tuple(ends))


def _require_recognition_result(candidate: ResultCandidate) -> RecognitionResult:
    required = ("text", "tokens", "timestamps", "durations", "lang")
    missing = [name for name in required if not hasattr(candidate, name)]
    assert not missing, f"binding result is missing required fields: {missing}"
    assert _is_recognition_result(candidate)
    return candidate


def _is_recognition_result(candidate: ResultCandidate) -> TypeGuard[RecognitionResult]:
    return all(
        hasattr(candidate, name)
        for name in ("text", "tokens", "timestamps", "durations", "lang")
    )


def test_mismatched_token_timestamps_are_rejected() -> None:
    # Given: a result whose binding fields cannot construct token events.
    result = FakeResult(
        text="Hello world.",
        tokens=("▁Hello", "▁world", "."),
        timestamps=(0.0, 0.5),
    )

    # When/Then: the compatibility boundary rejects, rather than invents, timing.
    with pytest.raises(AssertionError, match="cardinality mismatch"):
        _assert_result_contract(result, final_chunk_end_seconds=1.0)


@pytest.mark.parametrize(
    ("tokens", "timestamps"),
    [
        ((" A", "sk", " not", "."), (0.0, 0.1, 0.4, 0.7)),
        (("▁A", "sk", "▁", "not", "."), (0.0, 0.1, 0.4, 0.5, 0.7)),
    ],
)
def test_supported_word_boundaries_reconstruct_text(
    tokens: tuple[str, ...],
    timestamps: tuple[float, ...],
) -> None:
    # Given: equivalent word boundaries observed across supported tokenizers.
    result = FakeResult(
        text="Ask not.",
        tokens=tokens,
        timestamps=timestamps,
    )

    # When: the result crosses the compatibility boundary.
    observation = _assert_result_contract(result, final_chunk_end_seconds=0.8)

    # Then: both grammars produce the recognizer's exact normalized text.
    assert observation.reconstructed_text == "Ask not."


def test_absent_required_result_fields_are_rejected() -> None:
    # Given: an object lacking every binding result field.
    class MissingResult:
        pass

    # When/Then: the compatibility boundary names the absent fields.
    with pytest.raises(AssertionError, match="missing required fields"):
        _assert_result_contract(MissingResult(), final_chunk_end_seconds=1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_commit(bundle: ModelBundle) -> str:
    snapshot = bundle.encoder.parent
    assert snapshot.parent.name == "snapshots", (
        f"encoder is not inside a Hugging Face snapshot: {bundle.encoder}"
    )
    return snapshot.name


def _asset_identity(path: Path) -> Mapping[str, JsonValue]:
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _command_version(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return (completed.stdout or completed.stderr).splitlines()[0]


def _read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1, "integration WAV must be mono"
        assert stream.getsampwidth() == 2, "integration WAV must be 16-bit PCM"
        sample_rate = stream.getframerate()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0, sample_rate


def _environment_identity() -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "utc": datetime.now(UTC).isoformat(),
        "platform": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": sys.version,
        },
        "tools": {
            "poetry": _command_version(("poetry", "--version")),
            "ffmpeg": _command_version(("ffmpeg", "-version")),
        },
        "packages": versions,
    }


def _bundle_identity(bundle: ModelBundle, commit: str) -> dict[str, JsonValue]:
    paths = {
        "encoder": bundle.encoder,
        "decoder": bundle.decoder,
        "joiner": bundle.joiner,
        "tokens": bundle.tokens,
        "silero": bundle.silero,
    }
    return {
        "hugging_face": {
            "repo_id": PARAKEET_REPO_ID,
            "resolved_snapshot_commit": commit,
            "runtime_revision_pinned": False,
        },
        "runtime_assets": {
            name: _asset_identity(path) for name, path in paths.items()
        },
        "silero": {
            "source_url": SILERO_URL,
            "final_path": str(bundle.silero),
            "response_metadata_observable": False,
            "response_metadata": None,
        },
    }


@pytest.mark.integration
def test_current_binding_and_model_contract(
    tmp_path: Path,
    write_identity_json: IdentityWriter,
) -> None:
    wav_cache = tmp_path / "wav-cache"
    evidence: dict[str, JsonValue] = {
        "gate": {"verdict": "started", "assertion": None},
        "cleanup": {
            "wav_cache": str(wav_cache),
            "wav_cache_removed": False,
            "normalized_wav": None,
            "normalized_wav_removed": False,
        },
    }
    evidence.update(_environment_identity())
    failure: str | None = None
    prepared: PreparedAudio | None = None
    try:
        bundle = resolve_bundle()
        commit = _snapshot_commit(bundle)
        evidence.update(_bundle_identity(bundle, commit))
        wav = Path(
            hf_hub_download(
                repo_id=PARAKEET_REPO_ID,
                filename="test_wavs/en.wav",
                revision=commit,
                cache_dir=wav_cache,
            )
        )
        source_samples, source_sample_rate = _read_wave(wav)
        prepared = normalize_media(wav)
        samples, sample_rate = _read_wave(prepared.path)
        assert sample_rate == SAMPLE_RATE, f"integration WAV is {sample_rate} Hz"

        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(bundle.silero),
                threshold=0.5,
                min_silence_duration=0.5,
                min_speech_duration=0.25,
                window_size=512,
                max_speech_duration=20,
            ),
            sample_rate=SAMPLE_RATE,
            num_threads=1,
            provider="cpu",
            debug=False,
        )
        vad = sherpa_onnx.VoiceActivityDetector(
            vad_config,
            buffer_size_in_seconds=max(60.0, len(samples) / SAMPLE_RATE + 1),
        )
        vad.accept_waveform(samples)
        vad.flush()
        segments: list[VadSegmentEvidence] = []
        while not vad.empty():
            segment = vad.front
            segment_samples = np.asarray(segment.samples, dtype=np.float32)
            segments.append(
                {
                    "start": int(segment.start),
                    "start_seconds": int(segment.start) / SAMPLE_RATE,
                    "sample_count": len(segment_samples),
                    "chunk_end_seconds": (
                        int(segment.start) + len(segment_samples)
                    )
                    / SAMPLE_RATE,
                }
            )
            vad.pop()
        assert segments, "VAD returned no usable speech segments"
        assert all(segment["sample_count"] > 0 for segment in segments), (
            f"VAD returned empty segments: {segments}"
        )
        final_chunk_end = segments[-1]["chunk_end_seconds"]
        segments_json: list[JsonValue] = []
        for segment in segments:
            segments_json.append(
                {
                    "start": segment["start"],
                    "start_seconds": segment["start_seconds"],
                    "sample_count": segment["sample_count"],
                    "chunk_end_seconds": segment["chunk_end_seconds"],
                }
            )

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(bundle.encoder),
            decoder=str(bundle.decoder),
            joiner=str(bundle.joiner),
            tokens=str(bundle.tokens),
            num_threads=1,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
            model_type="nemo_transducer",
        )
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        recognizer.decode_stream(stream)
        result = _require_recognition_result(stream.result)
        fields = sorted(name for name in dir(result) if not name.startswith("_"))
        evidence["wav"] = {
            "repo_id": PARAKEET_REPO_ID,
            "path": str(wav),
            "sha256": _sha256(wav),
            "size": wav.stat().st_size,
            "source_sample_rate": source_sample_rate,
            "normalized_sample_rate": sample_rate,
            "channels": 1,
            "source_sample_count": len(source_samples),
            "normalized_sample_count": len(samples),
            "separate_runtime_asset": True,
        }
        evidence["vad"] = {
            "config": {
                "provider": "cpu",
                "sample_rate": SAMPLE_RATE,
                "window_size": 512,
            },
            "segments": segments_json,
            "usable_segment_count": len(segments),
        }
        result_evidence: dict[str, JsonValue] = {
            "field_names": fields,
            "cardinalities": {
                "tokens": len(result.tokens),
                "timestamps": len(result.timestamps),
                "durations": len(result.durations),
            },
            "timestamp_units": "seconds",
            "timestamps_monotonic": True,
            "raw_tokens": list(result.tokens),
            "raw_timestamps": list(result.timestamps),
            "raw_durations": list(result.durations),
            "lang": result.lang,
            "raw_text": result.text,
            "normalized_recognizer_text": _normalize(result.text),
            "grammar_verdict": "pending",
        }
        evidence["result"] = result_evidence
        observation = _assert_result_contract(
            result,
            final_chunk_end_seconds=final_chunk_end,
        )
        result_evidence.update(
            {
                "observed_control_tokens": list(observation.control_tokens),
                "token_end_seconds": list(observation.token_end_seconds),
                "reconstructed_text": observation.reconstructed_text,
                "normalized_reconstructed_text": _normalize(
                    observation.reconstructed_text
                ),
                "text_matches": True,
                "grammar_verdict": "pass",
            }
        )
        evidence["gate"] = {"verdict": "pass", "assertion": "all"}
    except Exception as error:  # noqa: BROAD_EXCEPT_OK
        failure = f"{type(error).__name__}: {error}"
        evidence["gate"] = {
            "verdict": "fail",
            "assertion": failure,
        }
    finally:
        normalized_wav = prepared.path if prepared is not None else None
        if prepared is not None:
            prepared.cleanup()
        shutil.rmtree(wav_cache, ignore_errors=True)
        evidence["cleanup"] = {
            "wav_cache": str(wav_cache),
            "wav_cache_removed": not wav_cache.exists(),
            "normalized_wav": (
                str(normalized_wav) if normalized_wav is not None else None
            ),
            "normalized_wav_removed": (
                normalized_wav is not None and not normalized_wav.exists()
            ),
        }
        write_identity_json(evidence)

    assert failure is None, failure
