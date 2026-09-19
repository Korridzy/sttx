"""Pure, stdlib-only signed observations; no runtime or native package imports.

Finite samples can alias a true timing lattice to one of its multiples. The
observed gcd is evidence about this sample, not universal timing semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias, TypedDict

JsonValue: TypeAlias = (
    None | bool | int | float | str | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
)
VARIANTS: Final = (0, 160, 480, 800)
RAW_KEYS: Final = frozenset({"tokens", "timestamps_us", "durations_us", "text", "lang"})
# Token timings live on the encoder frame grid, and int8 kernels resolve a frame
# boundary differently on different CPUs, so the same model shifts a few tokens by
# one or two frames between machines. The envelope therefore tolerates drift of at
# most two frames on a bounded fraction of tokens; token sequences and transcript
# text stay exact-match, which is what a real backend change moves.
ENCODER_FRAME_US: Final = 80_000
MAX_DRIFT_US: Final = 2 * ENCODER_FRAME_US
MATERIAL_DRIFT_US: Final = ENCODER_FRAME_US // 2
MATERIAL_DRIFT_PERCENT: Final = 20


class Diagnostic(TypedDict):
    code: str
    message: str


class RawObservation(TypedDict):
    tokens: list[str]
    timestamps_us: list[int]
    durations_us: list[int]
    text: str
    lang: str


class RunObservation(RawObservation):
    variant: int


class ProductionRun(RawObservation):
    probe: str
    segment_index: int
    chunk_index: int
    start_sample: int
    sample_count: int


class SegmentObservation(TypedDict):
    id: int
    start_us: int
    end_us: int
    text: str


class TranscriptObservation(TypedDict):
    probe: str
    language: str
    duration_us: int
    text: str
    segments: list[SegmentObservation]


class SignedPayload(TypedDict):
    runs: list[RunObservation]
    production_runs: list[ProductionRun]
    production_transcripts: list[TranscriptObservation]


@dataclass(frozen=True, slots=True)
class ObservationError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def quantize_us(seconds: float) -> int:
    try:
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
            raise ObservationError("seconds must be finite and nonnegative")
        scaled = seconds * 1_000_000
        if not math.isfinite(scaled):
            raise ObservationError("seconds overflow microsecond conversion")
        return 10 * round(round(scaled) / 10)
    except OverflowError as error:
        raise ObservationError("seconds overflow microsecond conversion") from error


def derive_quantum_us(timestamps_us: Sequence[JsonValue]) -> int:
    checked = [_integer(value, quantum=10) for value in timestamps_us]
    values = [value for value in checked if value != 0]
    if len(values) < 30 or len(set(values)) < 8:
        raise ObservationError("lattice needs 30 nonzero observations and 8 distinct values")
    quantum = math.gcd(*values)
    if quantum < 1_000:
        raise ObservationError("observed lattice gcd is below 1000us")
    return quantum


def _integer(value: JsonValue, *, quantum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value % quantum:
        raise ObservationError("expected nonnegative integer on the required grid")
    return value


def _string(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ObservationError("expected string")
    try:
        _ = value.encode("utf-8")
    except UnicodeError as error:
        raise ObservationError("string is not UTF-8 encodable") from error
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ObservationError("expected JSON array")
    return value


def _shape(value: JsonValue, keys: frozenset[str]) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ObservationError(f"expected exact keys: {sorted(keys)}")
    return value


def _unique_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ObservationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _raw(data: Mapping[str, JsonValue]) -> RawObservation:
    tokens = [_string(value) for value in _array(data["tokens"])]
    starts = [_integer(value, quantum=10) for value in _array(data["timestamps_us"])]
    durations = [_integer(value, quantum=10) for value in _array(data["durations_us"])]
    if len(tokens) != len(starts) or (durations and len(durations) != len(tokens)):
        raise ObservationError("raw result cardinality mismatch")
    if any(left > right for left, right in zip(starts, starts[1:])):
        raise ObservationError("nonmonotonic token starts")
    return {"tokens": tokens, "timestamps_us": starts, "durations_us": durations,
            "text": _string(data["text"]), "lang": _string(data["lang"])}


def _transcript(value: JsonValue) -> TranscriptObservation:
    data = _shape(value, frozenset({"probe", "language", "duration_us", "text", "segments"}))
    duration = _integer(data["duration_us"], quantum=10)
    segments: list[SegmentObservation] = []
    previous_start = 0
    for index, item in enumerate(_array(data["segments"])):
        segment = _shape(item, frozenset({"id", "start_us", "end_us", "text"}))
        identity = _integer(segment["id"])
        start = _integer(segment["start_us"], quantum=10)
        end = _integer(segment["end_us"], quantum=10)
        if identity != index or start < previous_start or not start <= end <= duration:
            raise ObservationError("invalid transcript segment identity/order/bounds")
        segments.append({"id": identity, "start_us": start, "end_us": end,
                         "text": _string(segment["text"])})
        previous_start = start
    return {"probe": _string(data["probe"]), "language": _string(data["language"]),
            "duration_us": duration, "text": _string(data["text"]), "segments": segments}


def load_payload(encoded: str | bytes) -> SignedPayload:
    """Parse exactly the signed three collections, never a candidate envelope.

    Production probes follow transcript order. Segment/chunk IDs are zero-based
    and contiguous; a continued chunk starts at the previous chunk's sample end.
    Coverage requirements (specific probes, speech, long chunk) belong to task 8.
    """
    try:
        text = encoded.decode("utf-8") if isinstance(encoded, bytes) else encoded
        decoded: JsonValue = json.loads(text, object_pairs_hook=_unique_pairs)
    except (ValueError, RecursionError) as error:
        raise ObservationError(f"invalid observation JSON: {error}") from error
    data = _shape(decoded, frozenset({"runs", "production_runs", "production_transcripts"}))
    runs: list[RunObservation] = []
    for value in _array(data["runs"]):
        run = _shape(value, RAW_KEYS | {"variant"})
        runs.append({"variant": _integer(run["variant"]), **_raw(run)})
    if tuple(run["variant"] for run in runs) != VARIANTS:
        raise ObservationError("expected ordered lattice variants 0/160/480/800")
    transcripts = [_transcript(value) for value in _array(data["production_transcripts"])]
    probes = [transcript["probe"] for transcript in transcripts]
    if not probes or any(not probe for probe in probes) or len(set(probes)) != len(probes):
        raise ObservationError("missing or duplicate transcript probes")
    production: list[ProductionRun] = []
    for value in _array(data["production_runs"]):
        run = _shape(value, RAW_KEYS | {
            "probe", "segment_index", "chunk_index", "start_sample", "sample_count"})
        production.append({"probe": _string(run["probe"]),
            "segment_index": _integer(run["segment_index"]), "chunk_index": _integer(run["chunk_index"]),
            "start_sample": _integer(run["start_sample"]), "sample_count": _integer(run["sample_count"]),
            **_raw(run)})
    _production_order(production, transcripts)
    return {"runs": runs, "production_runs": production, "production_transcripts": transcripts}


def _production_order(runs: list[ProductionRun], transcripts: list[TranscriptObservation]) -> None:
    probes = [transcript["probe"] for transcript in transcripts]
    previous: ProductionRun | None = None
    seen: list[str] = []
    for run in runs:
        probe = run["probe"]
        if probe not in probes or not 0 < run["sample_count"] <= 480_000:
            raise ObservationError("invalid production probe or chunk size")
        end_sample = run["start_sample"] + run["sample_count"]
        duration = transcripts[probes.index(probe)]["duration_us"]
        if end_sample * 1_000_000 > (duration + 5) * 16_000:
            raise ObservationError("production chunk exceeds quantized input duration")
        if previous is None or previous["probe"] != probe:
            if probe in seen or run["segment_index"] != 0 or run["chunk_index"] != 0:
                raise ObservationError("duplicate/out-of-order production probe or initial identity")
            seen.append(probe)
        else:
            previous_end = previous["start_sample"] + previous["sample_count"]
            if run["start_sample"] < previous_end:
                raise ObservationError("nonmonotonic production chunk bounds")
            if run["segment_index"] == previous["segment_index"]:
                if (run["chunk_index"] != previous["chunk_index"] + 1
                        or run["start_sample"] != previous_end or previous["sample_count"] != 480_000):
                    raise ObservationError("invalid continued chunk identity/bounds")
            elif run["segment_index"] != previous["segment_index"] + 1 or run["chunk_index"] != 0:
                raise ObservationError("invalid next segment identity")
        previous = run
    if seen != probes:
        raise ObservationError("production and transcript probe order/coverage differ")


def canonical_bytes(payload: SignedPayload | Mapping[str, JsonValue]) -> bytes:
    """Validate even caller-mutated typed mappings before signing; no normalization."""
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ObservationError(f"invalid signed payload: {error}") from error
    if load_payload(encoded) != payload:
        raise ObservationError("signed payload would require coercion")
    return encoded


def signature_sha256(payload: SignedPayload) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _time_drift(before: Sequence[int], after: Sequence[int], path: str) -> tuple[Diagnostic, ...]:
    if len(before) != len(after):
        return ({"code": "behavior.length", "message": path},)
    deltas = [abs(left - right) for left, right in zip(before, after, strict=True)]
    material_count = sum(delta >= MATERIAL_DRIFT_US for delta in deltas)
    allowance = max(1, (MATERIAL_DRIFT_PERCENT * len(deltas)) // 100)
    if any(delta > MAX_DRIFT_US for delta in deltas) or material_count > allowance:
        return ({"code": "behavior.timing", "message": path},)
    return ()


def compare_payloads(before: SignedPayload, after: SignedPayload) -> tuple[Diagnostic, ...]:
    """Return blocking behavior diagnostics; malformed payloads raise ObservationError.

    Callers distinguish candidate.invalid from baseline.invalid at their boundary.
    An empty tuple means comparable behavior, not complete native qualification.
    """
    _ = canonical_bytes(before)
    _ = canonical_bytes(after)
    diagnostics: list[Diagnostic] = []
    for name in ("runs", "production_runs"):
        left_runs = before[name]
        right_runs = after[name]
        if len(left_runs) != len(right_runs):
            diagnostics.append({"code": "behavior.length", "message": name})
            continue
        for index, (left, right) in enumerate(zip(left_runs, right_runs, strict=True)):
            path = f"{name}[{index}]"
            left_identity = {key: value for key, value in left.items()
                             if key not in {"timestamps_us", "durations_us"}}
            right_identity = {key: value for key, value in right.items()
                              if key not in {"timestamps_us", "durations_us"}}
            if left_identity != right_identity or bool(left["durations_us"]) != bool(right["durations_us"]):
                diagnostics.append({"code": "behavior.identity", "message": path})
                continue
            diagnostics.extend(_time_drift(
                left["timestamps_us"], right["timestamps_us"], path + ".timestamps_us"))
            diagnostics.extend(_time_drift(
                left["durations_us"], right["durations_us"], path + ".durations_us"))
    left_transcripts, right_transcripts = before["production_transcripts"], after["production_transcripts"]
    if len(left_transcripts) != len(right_transcripts):
        diagnostics.append({"code": "behavior.length", "message": "production_transcripts"})
        return tuple(diagnostics)
    for left, right in zip(left_transcripts, right_transcripts, strict=True):
        path = f"production_transcripts[{left['probe']}]"
        if ([(item["id"], item["text"]) for item in left["segments"]]
                != [(item["id"], item["text"]) for item in right["segments"]]
                or any(left[key] != right[key] for key in ("probe", "language", "text", "duration_us"))):
            diagnostics.append({"code": "behavior.identity", "message": path})
            continue
        left_bounds = [bound for item in left["segments"] for bound in (item["start_us"], item["end_us"])]
        right_bounds = [bound for item in right["segments"] for bound in (item["start_us"], item["end_us"])]
        diagnostics.extend(_time_drift(left_bounds, right_bounds, path + ".segment_bounds"))
    return tuple(diagnostics)
