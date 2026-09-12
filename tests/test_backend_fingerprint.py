from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.backend_qualification_helpers import ROOT, SCRIPT, SOURCES, checkout, invoke


def test_print_when_isolated_uses_only_stdlib() -> None:
    result = invoke(ROOT)

    assert result.returncode == 0, result.stderr
    assert re.fullmatch(r"[0-9a-f]{64}\n", result.stdout)
    assert result.stderr == ""


@pytest.mark.parametrize("source", SOURCES)
def test_fingerprint_when_existing_source_changes(tmp_path: Path, source: str) -> None:
    root = checkout(tmp_path)
    before = invoke(root)
    path = root / source
    _ = path.write_bytes(path.read_bytes() + b"\n# sensitivity probe\n")

    after = invoke(root)

    assert before.returncode == after.returncode == 0, after.stderr
    assert before.stdout != after.stdout


def test_fingerprint_when_future_gate_appears(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    gate = root / SOURCES[2]
    if gate.exists():
        gate.unlink()
    before = invoke(root)
    _ = gate.write_text("raise AssertionError('must never import native gate')\n")

    after = invoke(root)

    assert before.returncode == after.returncode == 0, after.stderr
    assert before.stdout != after.stdout


@pytest.mark.parametrize("mutation,changed", [
    ("order", False), ("metadata", False), ("dependency", True), ("contract", True),
])
def test_fingerprint_when_identity_inputs_change(tmp_path: Path, mutation: str, changed: bool) -> None:
    from sttx.backend_contract import PROBE_VERSION

    root = checkout(tmp_path)
    before = invoke(root)
    path = root / ("src/sttx/backend_contract.py" if mutation == "contract" else "pyproject.toml")
    text = path.read_text()
    replacements = {
        "order": ('"sherpa-onnx==1.13.6",\n    "sherpa-onnx-bin==1.13.6",',
                  '"sherpa-onnx-bin==1.13.6",\n    "sherpa-onnx==1.13.6",'),
        "metadata": ('version = "0.1.1"', 'version = "9.9.9"'),
        "dependency": ("numpy==2.4.6", "numpy==2.4.7"),
        "contract": (f"PROBE_VERSION: Final = {PROBE_VERSION}", f"PROBE_VERSION: Final = {PROBE_VERSION + 1}"),
    }
    old, new = replacements[mutation]
    assert old in text
    _ = path.write_text(text.replace(old, new))

    after = invoke(root)

    assert before.returncode == after.returncode == 0, after.stderr
    assert (before.stdout != after.stdout) is changed


def test_loading_when_checkout_package_is_poisoned(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    _ = (root / "src/sttx/__init__.py").write_text("raise RuntimeError('package imported')\n")
    driver = (
        "import runpy,sys; before=list(sys.path); "
        f"namespace=runpy.run_path({str(root / SCRIPT)!r}); namespace['fingerprint'](); "
        "assert sys.path==before; "
        "assert not any(n.split('.')[0] in {'sttx','numpy','sherpa_onnx','pytest'} for n in sys.modules)"
    )

    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", driver],
                            capture_output=True, text=True, timeout=10)

    assert result.returncode == 0, result.stderr


def test_fingerprint_when_canonical_identity_is_independently_hashed(tmp_path: Path) -> None:
    from sttx.backend_contract import contract_payload
    import tomllib

    root = checkout(tmp_path)
    manifest = tomllib.loads((root / "pyproject.toml").read_text())
    identity = {"dependencies": sorted(manifest["project"]["dependencies"]),
                "contract": contract_payload(),
                "sources": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            if (root / name).exists() else "absent" for name in SOURCES}}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()

    result = invoke(root)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("encoded", ['[true]', '["numpy==1", "numpy==2"]', '"numpy==1"',
    '["numpy>=1"]', '["numpy==1", 2]'])
def test_cli_when_dependency_boundary_is_invalid_rejects(tmp_path: Path, encoded: str) -> None:
    root = checkout(tmp_path)
    _ = (root / "pyproject.toml").write_text("[project]\ndependencies = " + encoded + "\n")

    result = invoke(root)

    assert result.returncode != 0
    assert result.stdout == "" and "Traceback" not in result.stderr


@pytest.mark.parametrize("flags", [("--bootstrap",), ("--accept-behavior-drift",),
    ("--check", "absent.json", "--bootstrap"), ("--check", "absent.json"), ("--unknown",)])
def test_cli_when_arguments_are_unsafe_rejects(flags: tuple[str, ...]) -> None:
    result = invoke(ROOT, *flags)

    assert result.returncode != 0
    assert result.stdout == "" and "Traceback" not in result.stderr


@pytest.mark.parametrize("flags", [(), ("--bootstrap",), ("--accept-behavior-drift",)])
@pytest.mark.parametrize("code", ["unknown", "candidate.invalid", "candidate.coverage_invalid",
    "comparison.fixture_invalid", "comparison.platform_invalid", "baseline.invalid"])
def test_promotion_switches_when_diagnostic_is_forbidden_never_bypass(tmp_path: Path,
        flags: tuple[str, ...], code: str) -> None:
    from tests.backend_qualification_helpers import candidates, check_pair, comparison

    first, second = candidates(tmp_path)
    first["baseline_comparison"] = comparison(code)

    status = check_pair(tmp_path, (first, second), *flags)

    assert status != 0


@pytest.mark.parametrize("replacement", [
    '"signature_sha256": "0", "signature_sha256":',
    '"repo": "wrong", "repo":', '"tokens": [], "tokens":',
])
def test_complete_record_when_duplicate_key_is_hidden_rejects(tmp_path: Path, replacement: str) -> None:
    from tests.backend_qualification_helpers import candidates

    _, record = candidates(tmp_path)
    key = replacement.split(":")[0] + ":"
    text = json.dumps(record).replace(key, replacement, 1)
    _ = (tmp_path / "duplicate.json").write_text(text)
    _ = (tmp_path / "second.json").write_text(json.dumps(record))

    result = invoke(tmp_path, "--check-promotion", "duplicate.json", "second.json", "--bootstrap")

    assert result.returncode != 0 and "duplicate JSON key" in result.stderr


def test_baseline_when_payload_and_signature_are_coordinately_tampered_rejects(tmp_path: Path) -> None:
    from tests.backend_qualification_helpers import candidates, payload
    from tests.integration.timing_lattice import JsonValue, signature_sha256

    _, record = candidates(tmp_path)
    changed = payload()
    changed["runs"][0]["text"] = "tampered"
    collections: dict[str, JsonValue] = json.loads(json.dumps(changed))
    record.update(collections)
    record["signature_sha256"] = signature_sha256(changed)
    record["record_kind"] = "qualified_backend"
    record["promotion"] = {"previous_fingerprint": None, "previous_anchor": None,
        "comparison_to_previous": record["baseline_comparison"], "decision": "bootstrap"}
    _ = (tmp_path / "tampered.json").write_text(json.dumps(record))

    result = invoke(tmp_path, "--check", "tampered.json")

    assert result.returncode != 0 and "anchor" in result.stderr


@pytest.mark.parametrize("alias", [False, True])
def test_promotion_when_both_runs_are_one_file_rejects(tmp_path: Path, alias: bool) -> None:
    from tests.backend_qualification_helpers import candidates

    _, record = candidates(tmp_path)
    path = tmp_path / "candidate.json"
    _ = path.write_text(json.dumps(record))
    second = tmp_path / "alias.json" if alias else path
    if alias:
        second.symlink_to(path)

    result = invoke(tmp_path, "--check-promotion", str(path), str(second), "--bootstrap")

    assert result.returncode != 0 and "distinct" in result.stderr


@pytest.mark.parametrize("decision,codes,allowed", [
    ("routine", [], True), ("routine", ["baseline.dependencies_stale"], True),
    ("accepted_behavior", ["behavior.identity"], True), ("accepted_behavior", [], False),
    ("routine", ["behavior.timing"], False), ("unknown", [], False),
])
def test_baseline_cli_when_later_promotion_provenance_is_checked(tmp_path: Path,
        decision: str, codes: list[str], allowed: bool) -> None:
    from tests.backend_qualification_helpers import candidates, comparison

    _, record = candidates(tmp_path)
    record["record_kind"] = "qualified_backend"
    record["baseline_comparison"] = comparison(*codes)
    record["promotion"] = {"previous_fingerprint": "1" * 64, "previous_anchor": "2" * 64,
        "comparison_to_previous": record["baseline_comparison"], "decision": decision}
    _ = (tmp_path / "baseline.json").write_text(json.dumps(record))

    result = invoke(tmp_path, "--check", "baseline.json")

    assert (result.returncode == 0) is allowed, result.stderr


def _overflow_records(root: Path, field: str) -> tuple[Path, Path]:
    from tests.backend_qualification_helpers import candidates, payload
    from tests.integration.timing_lattice import JsonValue

    records = candidates(root)
    signed = payload()
    if field == "duration":
        signed["production_transcripts"][0]["duration_us"] = 10**400
    else:
        signed["production_runs"][0]["timestamps_us"] = [10**400] * 8
    collections: dict[str, JsonValue] = json.loads(json.dumps(signed))
    paths = (root / "C1.json", root / "C2.json")
    for path, record in zip(paths, records, strict=True):
        record.update(collections)
        record["signature_sha256"] = hashlib.sha256(json.dumps(
            collections, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        _ = path.write_text(json.dumps(record))
    return paths


@pytest.mark.parametrize("field", ["duration", "timestamps"])
def test_load_record_when_observation_arithmetic_overflows_rejects(tmp_path: Path, field: str) -> None:
    from scripts import compute_backend_fingerprint as checker

    first, _ = _overflow_records(tmp_path, field)

    with pytest.raises(checker.lattice.ObservationError) as failure:
        checker.load_record(first, tmp_path)

    assert isinstance(failure.value.__cause__, OverflowError)


@pytest.mark.parametrize("field", ["duration", "timestamps"])
def test_cli_when_observation_arithmetic_overflows_rejects(tmp_path: Path, field: str) -> None:
    first, second = _overflow_records(tmp_path, field)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in tmp_path.rglob("*") if path.is_file()}

    result = invoke(tmp_path, "--check-promotion", str(first), str(second), "--bootstrap")

    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr.startswith("error: ") and "Traceback" not in result.stderr
    assert "observation arithmetic" in result.stderr
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("sample_count,overhang_us", [(16000, 0), (15999, 60),
    (15998, 120), (15997, 190), (15996, 250)])
def test_load_record_when_overhang_is_representable_preserves_quantization(tmp_path: Path,
        sample_count: int, overhang_us: int) -> None:
    from scripts import compute_backend_fingerprint as checker
    from tests.backend_qualification_helpers import candidates, payload
    from tests.integration.timing_lattice import JsonValue

    _, record = candidates(tmp_path)
    signed = payload()
    signed["production_runs"][0]["sample_count"] = sample_count
    signed["production_runs"][0]["timestamps_us"] = [1_000_000] * 8
    collections: dict[str, JsonValue] = json.loads(json.dumps(signed))
    record.update(collections)
    signature = hashlib.sha256(json.dumps(collections, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()
    record["signature_sha256"] = signature
    record["observed_max_overhang_us"] = overhang_us
    path = tmp_path / "candidate.json"
    _ = path.write_text(json.dumps(record))

    parsed = checker.load_record(path, tmp_path)

    assert parsed.signature == signature
    assert parsed.payload == signed


@pytest.mark.parametrize("source", SOURCES)
def test_qualification_when_any_bound_source_absent_rejects(tmp_path: Path, source: str) -> None:
    from scripts import compute_backend_fingerprint as checker

    root = checkout(tmp_path)
    (root / source).unlink()

    with pytest.raises(FileNotFoundError):
        checker.fingerprint(root, require_sources=True)


def test_manifest_when_active_has_exact_twenty_six_sources() -> None:
    from scripts import compute_backend_fingerprint as checker

    assert checker.SOURCES == SOURCES
    assert len(SOURCES) == len(set(SOURCES)) == 26
