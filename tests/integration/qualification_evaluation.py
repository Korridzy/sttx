from __future__ import annotations

from pathlib import Path

from tests.integration import qualification_schema as schema
from tests.integration import qualification_validation as validation
from tests.integration import timing_lattice as lattice


def evaluate_qualification(observations: schema.Observations, baseline: schema.BaselineLoadResult,
        *, expected_baseline_runs_sha256: str, candidate_output_path: Path,
        expected: schema.Expectations) -> schema.Candidate:
    comparison: schema.BaselineComparison = {"comparable": False, "integrity": list(baseline.diagnostics),
                                            "identity": [], "behavior": []}
    failures: tuple[lattice.Diagnostic, ...]
    try:
        failures = validation.validate_candidate(observations, expected)
    except (ValueError, RecursionError) as error:
        failures = ({"code": "candidate.invalid", "message": f"{candidate_output_path}: {error}"},)
    for failure in failures:
        group = "identity" if failure["code"] in schema.IDENTITY else "integrity"
        comparison[group].append(failure)
    previous = baseline.observations
    if previous is None:
        if not comparison["integrity"]:
            comparison["integrity"].append({"code": "baseline.invalid", "message": "missing parsed baseline"})
        return schema.candidate(observations, comparison)
    try:
        previous = validation.parse_observations(schema.json_value(previous))
        payload = schema.signed_payload(previous)
        validation.require(previous["signature_sha256"] == lattice.signature_sha256(payload), "baseline signature mismatch")
    except (ValueError, RecursionError) as error:
        comparison["integrity"].append({"code": "baseline.invalid", "message": str(error)})
        return schema.candidate(observations, comparison)
    deltas = (
        (previous["fingerprint"], expected.fingerprint, "baseline.fingerprint_stale"),
        (previous["probe_version"], expected.probe_version, "baseline.probe_version_stale"),
        (previous["model"]["repo"], expected.model["repo"], "baseline.model_repo_stale"),
        (previous["model"]["revision"], expected.model["revision"], "baseline.model_revision_stale"),
        (previous["model"]["sha256"], expected.model["sha256"], "baseline.model_assets_stale"),
        (previous["dependencies"], expected.dependencies, "baseline.dependencies_stale"),
        (previous["derived_quantum_us"], expected.quantum_us, "baseline.quantum_stale"),
        (previous["signature_sha256"], expected_baseline_runs_sha256, "baseline.anchor_stale"),
        (previous["fixture"], observations["fixture"], "comparison.fixture_invalid"),
        (previous["platform"], observations["platform"], "comparison.platform_invalid"),
    )
    comparison["identity"].extend({"code": code, "message": f"previous={before!r}; current={after!r}"}
                                  for before, after, code in deltas if before != after)
    blockers = {item["code"] for item in comparison["identity"]} - schema.STALE
    comparison["comparable"] = not failures and not comparison["integrity"] and not blockers
    if comparison["comparable"]:
        comparison["behavior"].extend(lattice.compare_payloads(payload, schema.signed_payload(observations)))
    return schema.candidate(observations, comparison)
