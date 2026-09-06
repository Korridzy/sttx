from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Final

import pytest

from tests.backend_qualification_helpers import SOURCES, candidates, check_pair, comparison, invoke, payload
from tests.integration.timing_lattice import JsonValue, signature_sha256

STALE: Final = ["baseline." + name + "_stale" for name in (
    "fingerprint", "probe_version", "model_repo", "model_revision", "model_assets", "dependencies", "quantum")]
FORBIDDEN: Final = ["candidate." + name + "_invalid" for name in (
    "quantum", "overhang", "model", "asset", "fixture", "dependencies", "coverage")]


@pytest.mark.parametrize("decision,codes,flags", [
    ("bootstrap", ["baseline.missing"], ("--bootstrap",)),
    ("routine", [], ()), ("routine", STALE, ()),
    ("accepted_behavior", ["behavior.timing"], ("--accept-behavior-drift",)),
])
def test_pair_when_policy_allows_is_read_only(tmp_path: Path, decision: str,
        codes: list[str], flags: tuple[str, ...]) -> None:
    first, second = candidates(tmp_path)
    first["baseline_comparison"] = comparison(*codes, comparable=decision != "bootstrap")
    second["baseline_comparison"] = comparison(*codes, comparable=decision != "bootstrap")

    status = check_pair(tmp_path, (first, second), *flags)

    assert status == 0


@pytest.mark.parametrize("code", [*FORBIDDEN, "candidate.invalid", "baseline.invalid",
    "candidate.lattice_inconclusive", "comparison.fixture_invalid", "comparison.platform_invalid",
    "unknown", "baseline.anchor_stale", "behavior.timing"])
def test_pair_when_diagnostic_is_forbidden_rejects(tmp_path: Path, code: str) -> None:
    first, second = candidates(tmp_path)
    first["baseline_comparison"] = comparison(code)
    second["baseline_comparison"] = comparison(code)

    status = check_pair(tmp_path, (first, second))

    assert status != 0


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("probe_version", True), ("derived_quantum_us", True),
    ("observed_max_overhang_us", False), ("observed_max_overhang_us", 1_000_010),
    ("inconclusive_reason", "insufficient"), ("model", {}), ("fixture", {}), ("platform", {}),
    ("dependencies", {}), ("runs", []), ("production_runs", []), ("production_transcripts", []),
    ("record_kind", "qualification_failure_envelope"), ("fingerprint", "0" * 64),
    ("signature_sha256", "0" * 64), ("baseline_comparison", {"comparable": True}),
])
def test_pair_when_candidate_is_invalid_despite_empty_diagnostics(tmp_path: Path,
        field: str, value: JsonValue) -> None:
    first, second = candidates(tmp_path)
    first["baseline_comparison"] = second["baseline_comparison"] = comparison()
    second[field] = value

    status = check_pair(tmp_path, (first, second), "--accept-behavior-drift")

    assert status != 0


@pytest.mark.parametrize("field", ["runs", "production_runs", "production_transcripts"])
def test_pair_when_signed_collection_changes_rejects(tmp_path: Path, field: str) -> None:
    first, second = candidates(tmp_path)
    changed = payload()
    changed[field][0]["text"] = "different"
    collections: dict[str, JsonValue] = json.loads(json.dumps(changed))
    second.update(collections)
    second["signature_sha256"] = signature_sha256(changed)

    status = check_pair(tmp_path, (first, second), "--bootstrap")

    assert status != 0


@pytest.mark.parametrize("mutation", ["anchor", "missing-source", "extra-key", "bootstrap-drift"])
def test_pair_when_transition_is_unsafe_rejects(tmp_path: Path, mutation: str) -> None:
    first, second = candidates(tmp_path)
    if mutation == "anchor":
        path = tmp_path / "src/sttx/backend_contract.py"
        _ = path.write_text(path.read_text().replace(str(first["signature_sha256"]), "0" * 64))
        second["fingerprint"] = invoke(tmp_path).stdout.strip()
    elif mutation == "missing-source":
        (tmp_path / SOURCES[2]).unlink()
        second["fingerprint"] = invoke(tmp_path).stdout.strip()
    elif mutation == "extra-key":
        second["unexpected"] = None
    flags = ("--bootstrap", "--accept-behavior-drift") if mutation == "bootstrap-drift" else ("--bootstrap",)

    status = check_pair(tmp_path, (first, second), *flags)

    assert status != 0


@pytest.mark.parametrize("mutation", ["valid", "kind", "provenance", "previous", "decision", "provenance-bool"])
def test_baseline_when_provenance_is_checked(tmp_path: Path, mutation: str) -> None:
    _, baseline = candidates(tmp_path)
    baseline["record_kind"] = "qualified_backend"
    promotion: dict[str, JsonValue] = {"previous_fingerprint": None, "previous_anchor": None,
        "comparison_to_previous": copy.deepcopy(baseline["baseline_comparison"]), "decision": "bootstrap"}
    baseline["promotion"] = promotion
    if mutation == "kind":
        baseline["record_kind"] = "qualification_candidate"
    elif mutation == "provenance":
        promotion["comparison_to_previous"] = comparison()
    elif mutation == "previous":
        promotion["previous_anchor"] = "0" * 64
    elif mutation == "decision":
        promotion["decision"] = "routine"
    elif mutation == "provenance-bool":
        malformed = comparison("baseline.missing", comparable=False)
        malformed["comparable"] = 0
        promotion["comparison_to_previous"] = malformed
    _ = (tmp_path / "baseline.json").write_text(json.dumps(baseline))

    result = invoke(tmp_path, "--check", "baseline.json")

    assert (result.returncode == 0) is (mutation == "valid"), result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("mutation", ["transcript-text", "long-length", "coverage", "lattice", "overhang"])
def test_candidate_when_resigned_invalid_observations_reject(tmp_path: Path, mutation: str) -> None:
    first, second = candidates(tmp_path)
    changed = payload()
    if mutation == "transcript-text":
        changed["production_transcripts"][0]["text"] = "not the segment text"
    elif mutation == "long-length":
        changed["production_transcripts"][1]["duration_us"] += 10
    elif mutation == "coverage":
        _ = changed["production_runs"].pop()
    elif mutation == "lattice":
        for run in changed["runs"]:
            run["timestamps_us"] = [0] * 8
    elif mutation == "overhang":
        changed["production_runs"][0]["timestamps_us"] = [2_000_010] * 8
    collections: dict[str, JsonValue] = json.loads(json.dumps(changed))
    for record in (first, second):
        record.update(collections)
        record["signature_sha256"] = signature_sha256(changed)
    path = tmp_path / "src/sttx/backend_contract.py"
    _ = path.write_text(path.read_text().replace(signature_sha256(payload()), signature_sha256(changed)))
    second["fingerprint"] = invoke(tmp_path).stdout.strip()

    status = check_pair(tmp_path, (first, second), "--bootstrap")

    assert status != 0


@pytest.mark.parametrize("encoded", [b"[]", b"null", b"\xff", b'{"runs": [], "runs": []}',
    b'{"nested":{"key":1,"key":2}}', b'{"schema_version":NaN}', b"[" * 2000])
def test_cli_when_json_is_malformed_rejects_without_traceback(tmp_path: Path, encoded: bytes) -> None:
    _ = candidates(tmp_path)
    _ = (tmp_path / "malformed.json").write_bytes(encoded)

    result = invoke(tmp_path, "--check", "malformed.json")

    assert result.returncode != 0
    assert result.stdout == "" and "Traceback" not in result.stderr


@pytest.mark.parametrize("mutation,allowed", [("anchor-stale", True), ("integrity", False),
    ("noncomparable", False), ("extra-identity", False), ("behavior", False)])
def test_c2_when_only_anchor_transition_is_allowed(tmp_path: Path, mutation: str, allowed: bool) -> None:
    first, second = candidates(tmp_path)
    first["baseline_comparison"] = comparison()
    cases = {
        "anchor-stale": comparison("baseline.anchor_stale", comparable=False),
        "integrity": comparison("baseline.missing", comparable=False),
        "noncomparable": comparison(comparable=False),
        "extra-identity": comparison("baseline.dependencies_stale"),
        "behavior": comparison("behavior.timing"),
    }
    second["baseline_comparison"] = cases[mutation]

    status = check_pair(tmp_path, (first, second))

    assert (status == 0) is allowed


@pytest.mark.parametrize("field", ["record_kind", "schema_version", "fingerprint", "probe_version", "model",
    "dependencies", "fixture", "platform", "derived_quantum_us", "inconclusive_reason",
    "observed_max_overhang_us", "runs", "production_runs", "production_transcripts", "signature_sha256",
    "baseline_comparison", "model.repo", "model.revision", "model.sha256", "fixture.name",
    "fixture.sha256", "platform.machine", "baseline_comparison.integrity", "baseline_comparison.behavior"])
def test_pair_when_required_field_is_missing_rejects(tmp_path: Path, field: str) -> None:
    first, second = candidates(tmp_path)
    components = field.split(".")
    node = second
    for component in components[:-1]:
        child = node[component]
        assert isinstance(child, dict)
        node = child
    del node[components[-1]]

    status = check_pair(tmp_path, (first, second), "--bootstrap")

    assert status != 0


def test_bootstrap_when_baseline_already_exists_rejects(tmp_path: Path) -> None:
    pair = candidates(tmp_path)
    _ = (tmp_path / "tests/integration/qualified_backend.json").write_text("null")

    status = check_pair(tmp_path, pair, "--bootstrap")

    assert status != 0
