from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

from sttx.backend_contract import contract_payload
from tests.integration import qualification_schema as schema
from tests.integration.timing_lattice import JsonValue, RawObservation, SignedPayload, signature_sha256

ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPT: Final = "scripts/compute_backend_fingerprint.py"
SOURCES: Final = (SCRIPT, "tests/integration/timing_lattice.py",
    "tests/integration/test_timing_qualification.py", "tests/conftest.py", "src/sttx/asr.py", "src/sttx/cli.py",
    "tests/integration/qualification_schema.py", "tests/integration/qualification_validation.py",
    "tests/integration/qualification_evaluation.py", "tests/integration/qualification_recording.py",
    "tests/integration/qualification_probes.py", "scripts/backend_change_detector.py")


def invoke(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", "-S", "-B", str(root / SCRIPT), *arguments],
                          capture_output=True, text=True, cwd=root, timeout=10)


def checkout(destination: Path) -> Path:
    for name in (*SOURCES, "src/sttx/backend_contract.py", "pyproject.toml"):
        source = ROOT / name
        if name == SOURCES[2] and not source.exists():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(source, target)
    return destination


def comparison(*codes: str, comparable: bool = True) -> dict[str, JsonValue]:
    return {"comparable": comparable, "integrity": [
        {"code": code, "message": "synthetic"} for code in codes if code == "baseline.missing"],
        "identity": [{"code": code, "message": "synthetic"} for code in codes
                     if code != "baseline.missing" and not code.startswith("behavior.")],
        "behavior": [{"code": code, "message": "synthetic"} for code in codes if code.startswith("behavior.")]}


def payload() -> SignedPayload:
    raw: RawObservation = {"tokens": ["word"] * 8, "timestamps_us": [80_000 * index for index in range(1, 9)],
           "durations_us": [], "text": "word", "lang": "en"}
    return {
        "runs": [{"variant": variant, **copy.deepcopy(raw)} for variant in (0, 160, 480, 800)],
        "production_runs": [
            {"probe": probe, "segment_index": 0, "chunk_index": chunk,
             "start_sample": start, "sample_count": count, **copy.deepcopy(raw)}
            for probe, chunk, start, count in (("original", 0, 0, 16000),
                ("long", 0, 0, 480000), ("long", 1, 480000, 96000))],
        "production_transcripts": [
            {"probe": probe, "language": "en", "duration_us": duration, "text": "word",
             "segments": [{"id": 0, "start_us": 0, "end_us": duration, "text": "word"}]}
            for probe, duration in (("original", 1_000_000), ("long", 36_000_000))],
    }


def candidates(root: Path) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    _ = checkout(root)
    path = root / "src/sttx/backend_contract.py"
    _ = path.write_text(re.sub(r'^QUALIFIED_RUNS_SHA256: Final = .+$',
        'QUALIFIED_RUNS_SHA256: Final = "unset"', path.read_text(), flags=re.MULTILINE))
    _ = (root / SOURCES[2]).write_text("raise RuntimeError('gate must not be imported')\n")
    settings = contract_payload()
    signed = payload()
    collections: dict[str, JsonValue] = json.loads(json.dumps(signed))
    first: dict[str, JsonValue] = {
        "record_kind": "qualification_candidate", "schema_version": 1,
        "fingerprint": invoke(root).stdout.strip(), "probe_version": settings["probe_version"],
        "model": {"repo": settings["parakeet_repo_id"], "revision": settings["parakeet_revision"],
                  "sha256": {**settings["parakeet_sha256"], settings["silero_filename"]: settings["silero_sha256"]}},
        "dependencies": {"sherpa-onnx": "1.13.6", "sherpa-onnx-bin": "1.13.6",
                         "huggingface-hub": "1.29.0", "numpy": "2.4.6"},
        "fixture": {"repo": settings["fixture_repo_id"], "revision": settings["fixture_revision"],
                    "name": settings["fixture_filename"], "sha256": settings["fixture_sha256"]},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "derived_quantum_us": 80_000, "inconclusive_reason": None, "observed_max_overhang_us": 0,
        **collections, "signature_sha256": signature_sha256(signed),
        "baseline_comparison": comparison("baseline.missing", comparable=False),
    }
    _ = path.write_text(path.read_text().replace('"unset"', json.dumps(first["signature_sha256"])))
    second = copy.deepcopy(first)
    second["fingerprint"] = invoke(root).stdout.strip()
    return first, second


def check_pair(root: Path, pair: tuple[dict[str, JsonValue], dict[str, JsonValue]], *flags: str) -> int:
    for name, record in zip(("C1.json", "C2.json"), pair, strict=True):
        _ = (root / name).write_text(json.dumps(record))
    paths = [path for path in root.rglob("*") if path.is_file()]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    result = invoke(root, "--check-promotion", "C1.json", "C2.json", *flags)

    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before
    assert result.stdout == "", result.stderr
    assert "Traceback" not in result.stderr
    return result.returncode


def observation_case() -> tuple[schema.Observations, schema.Expectations]:
    contract = contract_payload()
    expected = schema.Expectations(
        model={"repo": contract["parakeet_repo_id"], "revision": contract["parakeet_revision"],
            "sha256": {**contract["parakeet_sha256"], contract["silero_filename"]: contract["silero_sha256"]}},
        fixture={"repo": contract["fixture_repo_id"], "revision": contract["fixture_revision"],
            "name": contract["fixture_filename"], "sha256": contract["fixture_sha256"]},
        dependencies={"huggingface-hub": "1.29.0", "numpy": "2.4.6", "sherpa-onnx": "1.13.6",
                      "sherpa-onnx-bin": "1.13.6"},
        platform={"system": "Linux", "machine": "x86_64"}, probe_version=contract["probe_version"],
        quantum_us=80_000, max_overhang_us=1_000_000, fingerprint="f" * 64)
    signed = payload()
    observations: schema.Observations = {"record_kind": "qualification_candidate", "schema_version": 1,
        "fingerprint": expected.fingerprint, "probe_version": expected.probe_version,
        "model": copy.deepcopy(expected.model), "fixture": copy.deepcopy(expected.fixture),
        "dependencies": dict(expected.dependencies), "platform": copy.deepcopy(expected.platform),
        "derived_quantum_us": 80_000, "inconclusive_reason": None, "observed_max_overhang_us": 0,
        "signature_sha256": signature_sha256(signed), **signed}
    return observations, expected


def baseline_document(observations: schema.Observations) -> dict[str, JsonValue]:
    document = schema.document(observations)
    document["record_kind"] = "qualified_backend"
    previous = comparison("baseline.missing", comparable=False)
    document["baseline_comparison"] = previous
    document["promotion"] = {"previous_fingerprint": None, "previous_anchor": None,
        "comparison_to_previous": previous, "decision": "bootstrap"}
    return document
