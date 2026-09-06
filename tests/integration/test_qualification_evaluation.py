from __future__ import annotations

from pathlib import Path
import copy

import pytest

from tests.integration.qualification_schema import BaselineLoadResult
from tests.integration import qualification_schema as schema
from tests.integration.qualification_evaluation import evaluate_qualification
from tests.integration.timing_lattice import signature_sha256
from tests.backend_qualification_helpers import observation_case


def test_evaluation_when_baseline_missing_keeps_candidate_diagnostics(tmp_path: Path) -> None:
    from tests.integration.qualification_evaluation import evaluate_qualification
    from tests.backend_qualification_helpers import observation_case

    observations, expected = observation_case()
    observations["fixture"]["sha256"] = "0" * 64
    baseline = BaselineLoadResult(None, ({"code": "baseline.missing", "message": "absent"},))

    result = evaluate_qualification(observations, baseline,
        expected_baseline_runs_sha256="unset", candidate_output_path=tmp_path / "candidate.json", expected=expected)

    assert {item["code"] for item in result["baseline_comparison"]["integrity"]} == {
        "baseline.missing", "candidate.fixture_invalid"}
    assert result["baseline_comparison"]["comparable"] is False


def test_evaluation_when_stale_baseline_and_behavior_drift_reports_both(tmp_path: Path) -> None:
    observations, expected = observation_case()
    previous = copy.deepcopy(observations)
    previous["fingerprint"] = "a" * 64
    previous["dependencies"]["numpy"] = "2.4.5"
    previous["model"]["revision"] = "b" * 40
    previous["runs"][0]["text"] = "different"
    previous["signature_sha256"] = signature_sha256(schema.signed_payload(previous))

    result = evaluate_qualification(observations, BaselineLoadResult(previous, ()),
        expected_baseline_runs_sha256=previous["signature_sha256"],
        candidate_output_path=tmp_path / "candidate.json", expected=expected)["baseline_comparison"]

    assert result["comparable"]
    assert {item["code"] for item in result["identity"]} == {"baseline.fingerprint_stale",
        "baseline.dependencies_stale", "baseline.model_revision_stale"}
    assert {item["code"] for item in result["behavior"]} == {"behavior.identity"}
    assert result["integrity"] == []


@pytest.mark.parametrize("mutation,code", [("anchor", "baseline.anchor_stale"),
    ("fixture", "comparison.fixture_invalid"), ("platform", "comparison.platform_invalid")])
def test_evaluation_when_comparability_identity_differs_blocks(tmp_path: Path, mutation: str, code: str) -> None:
    observations, expected = observation_case()
    previous = copy.deepcopy(observations)
    anchor = previous["signature_sha256"]
    if mutation == "anchor":
        anchor = "0" * 64
    elif mutation == "fixture":
        previous["fixture"]["sha256"] = "0" * 64
    else:
        previous["platform"]["machine"] = "other"

    result = evaluate_qualification(observations, BaselineLoadResult(previous, ()),
        expected_baseline_runs_sha256=anchor, candidate_output_path=tmp_path / "candidate.json",
        expected=expected)["baseline_comparison"]

    assert not result["comparable"]
    assert [item["code"] for item in result["identity"]] == [code]
    assert result["behavior"] == []


@pytest.mark.parametrize("mutation,code", [("coverage", "candidate.coverage_invalid"),
    ("transcript", "candidate.coverage_invalid"), ("signature", "candidate.invalid"),
    ("quantum", "candidate.quantum_invalid"), ("overhang", "candidate.overhang_invalid"),
    ("model", "candidate.model_invalid"), ("asset", "candidate.asset_invalid"),
    ("dependencies", "candidate.dependencies_invalid"), ("platform", "comparison.platform_invalid")])
def test_evaluation_when_candidate_mutated_revalidates_before_comparison(tmp_path: Path,
        mutation: str, code: str) -> None:
    observations, expected = observation_case()
    previous = copy.deepcopy(observations)
    if mutation == "coverage":
        _ = observations["production_runs"].pop()
    elif mutation == "transcript":
        observations["production_transcripts"][0]["text"] = "not segment text"
    elif mutation == "quantum":
        observations["derived_quantum_us"] = 40000
    elif mutation == "overhang":
        observations["observed_max_overhang_us"] = 10
    elif mutation == "model":
        observations["model"]["repo"] = "wrong"
    elif mutation == "asset":
        observations["model"]["sha256"]["tokens.txt"] = "0" * 64
    elif mutation == "dependencies":
        observations["dependencies"]["numpy"] = "0"
    elif mutation == "platform":
        observations["platform"]["system"] = "Darwin"
    observations["signature_sha256"] = ("0" * 64 if mutation == "signature"
        else signature_sha256(schema.signed_payload(observations)))

    result = evaluate_qualification(observations, BaselineLoadResult(previous, ()),
        expected_baseline_runs_sha256=previous["signature_sha256"],
        candidate_output_path=tmp_path / "candidate.json", expected=expected)["baseline_comparison"]

    assert not result["comparable"]
    assert code in {item["code"] for item in result["integrity"] + result["identity"]}


def test_evaluation_when_complete_identical_payload_passes(tmp_path: Path) -> None:
    observations, expected = observation_case()

    result = evaluate_qualification(observations, BaselineLoadResult(copy.deepcopy(observations), ()),
        expected_baseline_runs_sha256=observations["signature_sha256"],
        candidate_output_path=tmp_path / "candidate.json", expected=expected)

    assert result["baseline_comparison"] == {"comparable": True, "integrity": [], "identity": [], "behavior": []}
