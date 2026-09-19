"""Strict parsing and candidate validity, independent of promotion decisions."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING or __package__:
    from tests.integration import qualification_schema as schema
    from tests.integration import timing_lattice as lattice


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise lattice.ObservationError(reason)


def digest(value: lattice.JsonValue) -> str:
    text = lattice._string(value)
    require(re.fullmatch("[0-9a-f]{64}", text) is not None, "invalid SHA-256")
    return text


def string_map(value: lattice.JsonValue) -> dict[str, str]:
    require(isinstance(value, dict), "expected string mapping")
    data = lattice._shape(value, frozenset(value) if isinstance(value, dict) else frozenset())
    return {lattice._string(key): lattice._string(item) for key, item in data.items()}


def parse_comparison(value: lattice.JsonValue) -> schema.BaselineComparison:
    data = lattice._shape(value, frozenset({"comparable", "integrity", "identity", "behavior"}))
    require(type(data["comparable"]) is bool, "comparable must be boolean")
    groups: list[list[lattice.Diagnostic]] = []
    for name, allowed in (("integrity", schema.INTEGRITY), ("identity", schema.IDENTITY),
                          ("behavior", schema.BEHAVIOR)):
        group: list[lattice.Diagnostic] = []
        for item in lattice._array(data[name]):
            diagnostic = lattice._shape(item, frozenset({"code", "message"}))
            code = lattice._string(diagnostic["code"])
            require(code in allowed, f"unknown {name} diagnostic: {code}")
            group.append({"code": code, "message": lattice._string(diagnostic["message"])})
        groups.append(group)
    return {"comparable": data["comparable"] is True, "integrity": groups[0],
            "identity": groups[1], "behavior": groups[2]}


def parse_observations(value: lattice.JsonValue) -> schema.Observations:
    data = lattice._shape(value, schema.OBSERVATION_KEYS)
    require(data["record_kind"] in ("qualification_candidate", "qualified_backend"), "incorrect record kind")
    require(lattice._integer(data["schema_version"]) == 1, "unsupported schema version")
    model = lattice._shape(data["model"], frozenset({"repo", "revision", "sha256"}))
    fixture = lattice._shape(data["fixture"], frozenset({"repo", "revision", "name", "sha256"}))
    platform = lattice._shape(data["platform"], frozenset({"system", "machine"}))
    assets = string_map(model["sha256"])
    require(bool(assets), "missing model assets")
    for hashed in assets.values():
        _ = digest(hashed)
    quantum = data["derived_quantum_us"]
    reason = data["inconclusive_reason"]
    payload = lattice.load_payload(schema.encode({name: data[name] for name in schema.COLLECTIONS}))
    return {"record_kind": lattice._string(data["record_kind"]), "schema_version": 1,
        "fingerprint": digest(data["fingerprint"]), "probe_version": lattice._integer(data["probe_version"]),
        "model": {"repo": lattice._string(model["repo"]), "revision": lattice._string(model["revision"]),
                  "sha256": assets},
        "fixture": {"repo": lattice._string(fixture["repo"]), "revision": lattice._string(fixture["revision"]),
                    "name": lattice._string(fixture["name"]), "sha256": digest(fixture["sha256"])},
        "platform": {"system": lattice._string(platform["system"]), "machine": lattice._string(platform["machine"])},
        "dependencies": string_map(data["dependencies"]),
        "derived_quantum_us": None if quantum is None else lattice._integer(quantum),
        "inconclusive_reason": None if reason is None else lattice._string(reason),
        "observed_max_overhang_us": lattice._integer(data["observed_max_overhang_us"]),
        "signature_sha256": digest(data["signature_sha256"]), **payload}


def parse_record(value: lattice.JsonValue, *, baseline: bool = False) -> schema.ParsedRecord:
    data = lattice._shape(value, schema.RECORD_KEYS | ({"promotion"} if baseline else set()))
    require(data["record_kind"] == ("qualified_backend" if baseline else "qualification_candidate"),
            "incorrect record kind")
    observations = parse_observations({name: data[name] for name in schema.OBSERVATION_KEYS})
    require(observations["signature_sha256"] == lattice.signature_sha256(schema.signed_payload(observations)),
            "signed payload mismatch")
    comparison = parse_comparison(data["baseline_comparison"])
    promotion: schema.Promotion | None = None
    if baseline:
        raw = lattice._shape(data["promotion"], frozenset({"previous_fingerprint", "previous_anchor",
                                                         "comparison_to_previous", "decision"}))
        prior = parse_comparison(raw["comparison_to_previous"])
        require(prior == comparison, "provenance comparison mismatch")
        decision = schema.Decision(lattice._string(raw["decision"]))
        previous_fingerprint, previous_anchor = raw["previous_fingerprint"], raw["previous_anchor"]
        promotion = {"previous_fingerprint": None if previous_fingerprint is None else digest(previous_fingerprint),
            "previous_anchor": None if previous_anchor is None else digest(previous_anchor),
            "comparison_to_previous": prior, "decision": decision}
        integrity = {item["code"] for item in prior["integrity"]}
        identity = {item["code"] for item in prior["identity"]}
        match decision:
            case schema.Decision.BOOTSTRAP:
                require(previous_anchor is None and previous_fingerprint is None and not prior["comparable"]
                        and integrity == {"baseline.missing"} and not identity and not prior["behavior"],
                        "invalid bootstrap provenance")
            case schema.Decision.ROUTINE | schema.Decision.ACCEPTED_BEHAVIOR:
                require(previous_anchor is not None and previous_fingerprint is not None
                        and prior["comparable"] and not integrity and identity <= schema.STALE
                        and bool(prior["behavior"]) == (decision is schema.Decision.ACCEPTED_BEHAVIOR), "misleading decision")
            case unreachable:
                assert_never(unreachable)
    return schema.ParsedRecord(observations, comparison, promotion)


def read_record(path: Path, *, baseline: bool = False) -> schema.ParsedRecord:
    decoded: lattice.JsonValue = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=lattice._unique_pairs)
    return parse_record(decoded, baseline=baseline)


def load_baseline(path: Path) -> schema.BaselineLoadResult:
    try:
        record = read_record(path, baseline=True)
        observations = record.observations
        payload = schema.signed_payload(observations)
        require(coverage_valid(payload), "baseline lacks production coverage")
        require(observations["inconclusive_reason"] is None, "inconclusive baseline")
        require(lattice.derive_quantum_us([time for run in payload["runs"] for time in run["timestamps_us"]])
                == observations["derived_quantum_us"], "baseline quantum mismatch")
        require(max_overhang(payload) == observations["observed_max_overhang_us"], "baseline overhang mismatch")
        return schema.BaselineLoadResult(observations, ())
    except FileNotFoundError as error:
        code = "baseline.invalid" if path.is_symlink() else "baseline.missing"
        return schema.BaselineLoadResult(None, ({"code": code, "message": str(error)},))
    except (OSError, ValueError, RecursionError) as error:
        return schema.BaselineLoadResult(None, ({"code": "baseline.invalid", "message": str(error)},))


def max_overhang(payload: lattice.SignedPayload) -> int:
    overhang = 0
    try:
        observations: list[tuple[lattice.RawObservation, float]] = [
            (run, payload["production_transcripts"][0]["duration_us"] + run["variant"] * 1_000_000 / 16_000)
            for run in payload["runs"]]
        observations.extend((run, run["sample_count"] * 1_000_000 / 16_000) for run in payload["production_runs"])
        for run, duration in observations:
            require(bool(run["tokens"]) and bool(run["text"].strip()), "missing raw speech coverage")
            ends = [start + (run["durations_us"][index] if run["durations_us"] else 0)
                    for index, start in enumerate(run["timestamps_us"])]
            overhang = max(overhang, lattice.quantize_us(max(0, max(ends) - duration) / 1_000_000))
    except OverflowError as error:
        raise lattice.ObservationError("observation arithmetic exceeds supported numeric magnitude") from error
    return overhang


def coverage_valid(payload: lattice.SignedPayload) -> bool:
    transcripts = payload["production_transcripts"]
    return ([item["probe"] for item in transcripts] == ["original", "long"]
        and transcripts[-1]["duration_us"] == 36_000_000
        and all(item["text"].strip() and item["segments"] for item in transcripts)
        and all(item["text"] == " ".join(" ".join(segment["text"].split())
                for segment in item["segments"] if segment["text"].strip()) for item in transcripts)
        and any(run["probe"] == "long" and run["chunk_index"] == 1 for run in payload["production_runs"]))


def validate_candidate(observations: schema.Observations, expected: schema.Expectations,
                       *, check_fingerprint: bool = True) -> tuple[lattice.Diagnostic, ...]:
    data = parse_observations(schema.json_value(observations))
    payload = schema.signed_payload(data)
    diagnostics: list[lattice.Diagnostic] = []
    checks = (
        (data["signature_sha256"] == lattice.signature_sha256(payload), "candidate.invalid", "signed payload mismatch"),
        (not check_fingerprint or data["fingerprint"] == expected.fingerprint, "candidate.invalid", "fingerprint mismatch"),
        (data["probe_version"] == expected.probe_version, "candidate.invalid", "candidate probe version"),
        (data["platform"] == expected.platform and expected.platform["system"] == "Linux",
         "comparison.platform_invalid", "unsupported or mismatched candidate platform"),
        (data["model"]["repo"] == expected.model["repo"] and data["model"]["revision"] == expected.model["revision"],
         "candidate.model_invalid", "candidate model differs from current contract"),
        (data["model"]["sha256"] == expected.model["sha256"], "candidate.asset_invalid", "model asset hash mismatch"),
        (data["fixture"] == expected.fixture, "candidate.fixture_invalid", "fixture identity mismatch"),
        (data["dependencies"] == expected.dependencies, "candidate.dependencies_invalid", "dependency identity mismatch"),
    )
    diagnostics.extend({"code": code, "message": reason} for valid, code, reason in checks if not valid)
    try:
        quantum = lattice.derive_quantum_us([time for run in payload["runs"] for time in run["timestamps_us"]])
        if data["derived_quantum_us"] != quantum or quantum != expected.quantum_us:
            diagnostics.append({"code": "candidate.quantum_invalid", "message": "invalid candidate quantum"})
    except lattice.ObservationError as error:
        diagnostics.append({"code": "candidate.lattice_inconclusive", "message": str(error)})
    if data["inconclusive_reason"] is not None:
        diagnostics.append({"code": "candidate.lattice_inconclusive", "message": data["inconclusive_reason"]})
    if not coverage_valid(payload):
        diagnostics.append({"code": "candidate.coverage_invalid", "message": "missing production speech/multichunk coverage"})
    overhang = max_overhang(payload)
    if data["observed_max_overhang_us"] != overhang or overhang > expected.max_overhang_us:
        diagnostics.append({"code": "candidate.overhang_invalid", "message": "candidate overhang mismatch or excess"})
    return tuple(diagnostics)
