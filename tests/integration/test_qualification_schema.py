from __future__ import annotations

from pathlib import Path
import json
from typing import Literal, assert_never

import pytest

from tests.backend_qualification_helpers import baseline_document, observation_case
from tests.integration.qualification_schema import OBSERVATION_KEYS, document
from tests.integration.qualification_validation import load_baseline, parse_observations
from tests.integration.timing_lattice import JsonValue, ObservationError


@pytest.mark.parametrize("encoded", [b"null", b"[]", b"\xff", b"{", b'{"x":1,"x":2}', b"[" * 2000])
def test_baseline_when_malformed_returns_diagnostic(tmp_path: Path, encoded: bytes) -> None:
    from tests.integration.qualification_validation import load_baseline

    path = tmp_path / "baseline.json"
    _ = path.write_bytes(encoded)

    result = load_baseline(path)

    assert result.observations is None
    assert [item["code"] for item in result.diagnostics] == ["baseline.invalid"]


def test_baseline_when_missing_returns_missing(tmp_path: Path) -> None:
    from tests.integration.qualification_validation import load_baseline

    result = load_baseline(tmp_path / "absent.json")

    assert result.observations is None
    assert [item["code"] for item in result.diagnostics] == ["baseline.missing"]


def test_baseline_when_directory_returns_io_diagnostic(tmp_path: Path) -> None:
    from tests.integration.qualification_validation import load_baseline

    result = load_baseline(tmp_path)

    assert result.observations is None
    assert result.diagnostics[0]["code"] == "baseline.invalid"


@pytest.mark.parametrize("field,value", [("schema_version", True), ("probe_version", False),
    ("model", []), ("platform", {"system": "Linux"}), ("derived_quantum_us", True),
    ("dependencies", {"numpy": True}), ("observed_max_overhang_us", False), ("runs", []),
    ("signature_sha256", "0" * 64), ("promotion", {}), ("record_kind", "qualification_failure_envelope")])
def test_baseline_when_record_invalid_is_nonthrowing(tmp_path: Path, field: str, value: JsonValue) -> None:
    observations, _ = observation_case()
    record = baseline_document(observations)
    record[field] = value
    path = tmp_path / "baseline.json"
    _ = path.write_text(json.dumps(record))

    result = load_baseline(path)

    assert result.observations is None
    assert [item["code"] for item in result.diagnostics] == ["baseline.invalid"]


@pytest.mark.parametrize("field", sorted(OBSERVATION_KEYS))
def test_observations_when_required_key_missing_rejects(field: str) -> None:
    observations, _ = observation_case()
    encoded = document(observations)
    del encoded[field]

    with pytest.raises(ObservationError):
        parse_observations(encoded)


def test_baseline_when_valid_but_stale_identity_remains_loadable(tmp_path: Path) -> None:
    observations, _ = observation_case()
    observations["fingerprint"] = "a" * 64
    observations["model"]["revision"] = "b" * 40
    observations["dependencies"]["numpy"] = "2.4.5"
    path = tmp_path / "baseline.json"
    _ = path.write_text(json.dumps(baseline_document(observations)))

    result = load_baseline(path)

    assert result.diagnostics == ()
    assert result.observations is not None
    assert result.observations["model"] == observations["model"]


@pytest.mark.parametrize("replacement", ['"repo": "wrong", "repo":', '"tokens": [], "tokens":',
                                           '"comparable": true, "comparable":'])
def test_baseline_when_duplicate_nested_keys_rejects(tmp_path: Path, replacement: str) -> None:
    observations, _ = observation_case()
    key = replacement.split(":")[0] + ":"
    text = json.dumps(baseline_document(observations)).replace(key, replacement, 1)
    path = tmp_path / "baseline.json"
    _ = path.write_text(text)

    result = load_baseline(path)

    assert result.observations is None
    assert result.diagnostics[0]["code"] == "baseline.invalid"


def test_baseline_when_historical_coverage_missing_rejects(tmp_path: Path) -> None:
    from tests.integration.timing_lattice import signature_sha256
    from tests.integration.qualification_schema import signed_payload

    observations, _ = observation_case()
    _ = observations["production_runs"].pop()
    observations["signature_sha256"] = signature_sha256(signed_payload(observations))
    path = tmp_path / "baseline.json"
    _ = path.write_text(json.dumps(baseline_document(observations)))

    result = load_baseline(path)

    assert result.observations is None
    assert result.diagnostics[0]["code"] == "baseline.invalid"


@pytest.mark.parametrize("field,value", [
    ("record_kind", "qualification_candidate"), ("schema_version", True),
    ("unexpected", None), ("fingerprint", "0" * 64), ("model", {}),
    ("dependencies", {}), ("fixture", {}), ("promotion", {}),
])
def test_real_baseline_when_record_contract_tampered_rejects(
    tmp_path: Path, field: str, value: JsonValue,
) -> None:
    from scripts import compute_backend_fingerprint as checker
    from tests.integration.qualification_schema import document
    from tests.integration.qualification_validation import read_record

    baseline = checker.ROOT / "tests/integration/qualified_backend.json"
    parsed = read_record(baseline, baseline=True)
    record = baseline_document(parsed.observations)
    record["baseline_comparison"] = document(parsed.comparison)
    assert parsed.promotion is not None
    record["promotion"] = document(parsed.promotion)
    record[field] = value
    path = tmp_path / "tampered.json"
    _ = path.write_text(json.dumps(record))

    with pytest.raises(ValueError):
        checker.check_current(checker.load_record(path, baseline=True))


@pytest.mark.parametrize("collection", ["runs", "production_runs", "production_transcripts"])
def test_real_baseline_when_collection_and_signature_tampered_rejects_at_anchor(
    tmp_path: Path, collection: Literal["runs", "production_runs", "production_transcripts"],
) -> None:
    from scripts import compute_backend_fingerprint as checker
    from tests.integration.qualification_schema import signed_payload
    from tests.integration.qualification_validation import read_record, validate_candidate
    from tests.integration.timing_lattice import signature_sha256

    parsed = read_record(checker.ROOT / "tests/integration/qualified_backend.json", baseline=True)
    observations = parsed.observations
    match collection:
        case "runs":
            observations["runs"][0]["text"] = "changed raw text"
        case "production_runs":
            observations["production_runs"][0]["text"] = "changed raw text"
        case "production_transcripts":
            transcript = observations["production_transcripts"][0]
            transcript["segments"][0]["text"] = transcript["text"] = "changed transcript"
        case unreachable:
            assert_never(unreachable)
    observations["signature_sha256"] = signature_sha256(signed_payload(observations))
    assert validate_candidate(observations, checker.expectations()) == ()
    path = tmp_path / "resigned.json"
    _ = path.write_text(json.dumps(baseline_document(observations)))
    record = checker.load_record(path, baseline=True)

    with pytest.raises(ValueError, match="current anchor mismatch"):
        checker.check_current(record)
