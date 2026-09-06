from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.compute_backend_fingerprint import ROOT, expectations, fingerprint
from sttx import backend_contract as contract
from tests.conftest import JsonValue, write_json
from tests.integration import qualification_probes as probes
from tests.integration import qualification_schema as schema
from tests.integration.qualification_evaluation import evaluate_qualification
from tests.integration.qualification_validation import load_baseline, require


def safe_output(output: Path, baseline: Path) -> Path:
    resolved = output.resolve()
    require(resolved != baseline.resolve(), "qualification output aliases committed baseline")
    if output.exists() and baseline.exists():
        require(not output.samefile(baseline), "qualification output aliases committed baseline")
    return resolved


def run_qualification(output: Path, scratch: Path,
                      baseline_path: Path = ROOT / "tests/integration/qualified_backend.json") -> schema.Candidate:
    destination = safe_output(output, baseline_path)
    expected = expectations()
    envelope: dict[str, JsonValue] = {"record_kind": "qualification_failure_envelope", "schema_version": 1,
        "phase": "started", "probe_version": contract.PROBE_VERSION,
        "fingerprint": fingerprint(require_sources=True), "started_at": datetime.now(UTC).isoformat()}
    write_json(destination, envelope)
    candidate_written = False
    try:
        baseline = load_baseline(baseline_path)
        with probes.collect_observations(scratch, expected) as observations:
            candidate = evaluate_qualification(observations, baseline,
                expected_baseline_runs_sha256=contract.QUALIFIED_RUNS_SHA256,
                candidate_output_path=destination, expected=expected)
            write_json(destination, schema.document(candidate))
            candidate_written = True
            comparison = candidate["baseline_comparison"]
            failures = comparison["integrity"] + comparison["identity"] + comparison["behavior"]
            assert not failures, schema.encode(comparison)
            return candidate
    except BaseException as error:  # noqa: BROAD_EXCEPT_OK - preserve evidence for interruption, then re-raise
        if not candidate_written:
            envelope["phase"] = "failed"
            envelope["exception"] = {"type": type(error).__name__, "message": str(error)}
            try:
                write_json(destination, envelope)
            except (OSError, ValueError) as artifact_error:
                error.add_note(f"qualification failure artifact could not be updated: {artifact_error}")
        raise


@pytest.mark.integration
def test_timing_qualification(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    requested = request.config.getoption("--qualification-output")
    output = requested if isinstance(requested, Path) else tmp_path / "qualification.json"
    _ = run_qualification(output, tmp_path)
