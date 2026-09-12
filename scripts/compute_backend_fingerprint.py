"""Read-only qualification identity. Print permits an absent future gate only.

Run with Python 3.11+ (including -I -S); no installed distribution is required.
Qualification checks require all twenty-six sources. Only trusted pure helpers
and the contract are executed; historical/base modules are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, assert_never

if TYPE_CHECKING:
    from sttx.backend_contract import ContractPayload
    from tests.integration import timing_lattice as lattice
    from tests.integration import qualification_schema as schema
    from tests.integration import qualification_validation as validation
    from tests.integration.timing_lattice import JsonValue, SignedPayload

ROOT: Final = Path(__file__).resolve().parents[1]
SOURCES: Final = ("scripts/compute_backend_fingerprint.py", "tests/integration/timing_lattice.py",
    "tests/integration/test_timing_qualification.py", "tests/conftest.py",
    "src/sttx/asr.py", "src/sttx/cli.py",
    "tests/integration/qualification_schema.py", "tests/integration/qualification_validation.py",
    "tests/integration/qualification_evaluation.py", "tests/integration/qualification_recording.py",
    "tests/integration/qualification_probes.py", "scripts/backend_change_detector.py",
    "src/sttx/model.py", "src/sttx/audio.py", "src/sttx/asr_events.py",
    "tests/integration/test_binding_contract.py", "tests/integration/test_real_pipeline.py",
    "tests/integration/real_pipeline_artifacts.py", "tests/integration/real_pipeline_checks.py",
    "tests/integration/real_pipeline_evidence_contract.py", "tests/integration/real_pipeline_media.py",
    "tests/integration/real_pipeline_observability.py", "tests/integration/real_pipeline_runner.py",
    "tests/integration/real_pipeline_signal_process.py", "tests/integration/real_pipeline_signals.py",
    "tests/integration/real_pipeline_trace.py")


def _load(path: Path, **bindings: ModuleType) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_sttx_qualification_" + path.stem, path)
    if spec is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(bindings)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    finally:
        if previous is None:
            del sys.modules[spec.name]
        else:
            sys.modules[spec.name] = previous
    return module


def contract_payload(root: Path = ROOT) -> ContractPayload:
    factory: Callable[[], ContractPayload] = _load(root / "src/sttx/backend_contract.py").contract_payload
    return factory()


def dependencies(root: Path = ROOT) -> list[str]:
    with (root / "pyproject.toml").open("rb") as stream:
        project: JsonValue = tomllib.load(stream).get("project", {})
    if not isinstance(project, dict):
        raise lattice.ObservationError("missing PEP621 project table")
    values = [lattice._string(item) for item in lattice._array(project.get("dependencies"))]
    _require(bool(values) and all(re.fullmatch(r"[a-z0-9][a-z0-9-]*==[0-9][a-zA-Z0-9.!+_-]*", item)
                                  for item in values), "dependencies must be canonical exact pins")
    _require(len({item.split("==")[0] for item in values}) == len(values), "duplicate dependency name")
    return sorted(values)


def fingerprint(root: Path = ROOT, *, require_sources: bool = False) -> str:
    sources: dict[str, str] = {}
    for name in SOURCES:
        path = root / name
        if not path.exists() and not require_sources and name == SOURCES[2]:
            sources[name] = "absent"
        else:
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {"dependencies": dependencies(root), "contract": contract_payload(root), "sources": sources}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


if not TYPE_CHECKING:
    lattice = _load(ROOT / "tests/integration/timing_lattice.py")
    schema = _load(ROOT / "tests/integration/qualification_schema.py", lattice=lattice)
    validation = _load(ROOT / "tests/integration/qualification_validation.py", lattice=lattice, schema=schema)

STALE: Final = schema.STALE
BEHAVIOR: Final = schema.BEHAVIOR
COLLECTIONS: Final = schema.COLLECTIONS
RECORD_KEYS: Final = schema.RECORD_KEYS


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise lattice.ObservationError(reason)


def _digest(value: JsonValue) -> str:
    text = lattice._string(value)
    _require(re.fullmatch("[0-9a-f]{64}", text) is not None, "invalid SHA-256")
    return text


@dataclass(frozen=True, slots=True)
class Comparison:
    comparable: bool
    integrity: frozenset[str]
    identity: frozenset[str]
    behavior: frozenset[str]


def load_comparison(value: JsonValue) -> Comparison:
    data = lattice._shape(value, frozenset({"comparable", "integrity", "identity", "behavior"}))
    comparable = data["comparable"]
    _require(type(comparable) is bool, "comparable must be boolean")
    groups: list[frozenset[str]] = []
    for name, allowed in (("integrity", {"baseline.missing"}),
                          ("identity", STALE | {"baseline.anchor_stale"}), ("behavior", BEHAVIOR)):
        codes: list[str] = []
        for item in lattice._array(data[name]):
            diagnostic = lattice._shape(item, frozenset({"code", "message"}))
            code = lattice._string(diagnostic["code"])
            _ = lattice._string(diagnostic["message"])
            _require(code in allowed, f"forbidden or unknown {name} diagnostic: {code}")
            codes.append(code)
        groups.append(frozenset(codes))
    return Comparison(comparable is True, *groups)


class Decision(StrEnum):
    BOOTSTRAP = "bootstrap"
    ROUTINE = "routine"
    ACCEPTED_BEHAVIOR = "accepted_behavior"


def _policy(comparison: Comparison, decision: Decision) -> None:
    match decision:
        case Decision.BOOTSTRAP:
            _require(comparison == Comparison(False, frozenset({"baseline.missing"}), frozenset(), frozenset()),
                     "bootstrap requires only baseline.missing")
        case Decision.ROUTINE | Decision.ACCEPTED_BEHAVIOR:
            _require(not comparison.integrity and comparison.identity <= STALE and comparison.comparable,
                     "routine promotion requires comparable C1 with empty integrity")
            _require(not comparison.behavior or decision is Decision.ACCEPTED_BEHAVIOR,
                     "behavior drift requires explicit acceptance")
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class Record:
    fingerprint: str
    signature: str
    payload: SignedPayload
    comparison: Comparison


def load_record(path: Path, root: Path = ROOT, *, baseline: bool = False) -> Record:
    """Parse a complete promotable record; never trust its supplied diagnostics."""
    parsed = validation.read_record(path, baseline=baseline)
    data = parsed.observations
    diagnostics = validation.validate_candidate(data, expectations(root), check_fingerprint=False)
    _require(not diagnostics, "; ".join(item["message"] for item in diagnostics))
    payload = schema.signed_payload(data)
    signature = data["signature_sha256"]
    comparison = load_comparison(schema.json_value(parsed.comparison))
    if baseline:
        promotion = parsed.promotion
        assert promotion is not None
        _ = load_comparison(schema.json_value(promotion["comparison_to_previous"]))
        decision = Decision(promotion["decision"])
        _policy(comparison, decision)
        match decision:
            case Decision.BOOTSTRAP:
                _require(promotion["previous_fingerprint"] is None and promotion["previous_anchor"] is None,
                         "bootstrap previous identity must be null")
            case Decision.ROUTINE | Decision.ACCEPTED_BEHAVIOR:
                _ = _digest(promotion["previous_fingerprint"])
                _ = _digest(promotion["previous_anchor"])
                _require(bool(comparison.behavior) == (decision is Decision.ACCEPTED_BEHAVIOR), "misleading decision")
            case unreachable:
                assert_never(unreachable)
    return Record(_digest(data["fingerprint"]), signature, payload, comparison)


def expectations(root: Path = ROOT) -> schema.Expectations:
    contract = contract_payload(root)
    return schema.Expectations(
        model={"repo": contract["parakeet_repo_id"], "revision": contract["parakeet_revision"],
               "sha256": {**contract["parakeet_sha256"], contract["silero_filename"]: contract["silero_sha256"]}},
        fixture={"repo": contract["fixture_repo_id"], "revision": contract["fixture_revision"],
                 "name": contract["fixture_filename"], "sha256": contract["fixture_sha256"]},
        dependencies=dict(item.split("==") for item in dependencies(root)),
        platform={"system": platform.system(), "machine": platform.machine()},
        probe_version=contract["probe_version"], quantum_us=contract["encoder_frame_us"],
        max_overhang_us=contract["max_token_overhang_us"], fingerprint=fingerprint(root))


def check_current(record: Record, root: Path = ROOT) -> None:
    _require(record.fingerprint == fingerprint(root, require_sources=True), "current fingerprint mismatch")
    _require(record.signature == contract_payload(root)["qualified_runs_sha256"], "current anchor mismatch")


def check_promotion(paths: tuple[Path, Path], root: Path = ROOT, *, decision: Decision = Decision.ROUTINE) -> None:
    bootstrap = decision is Decision.BOOTSTRAP
    _require(not paths[0].samefile(paths[1]), "promotion requires two distinct candidate files")
    baseline_path = root / "tests/integration/qualified_backend.json"
    _require(not bootstrap or not (baseline_path.exists() or baseline_path.is_symlink()), "bootstrap baseline exists")
    first, second = (load_record(path, root) for path in paths)
    _policy(first.comparison, decision)
    check_current(second, root)
    _require(lattice.canonical_bytes(first.payload) == lattice.canonical_bytes(second.payload), "unstable signed pair")
    if bootstrap:
        _policy(second.comparison, Decision.BOOTSTRAP)
    else:
        later = second.comparison
        _require(not later.integrity and later.identity - {"baseline.anchor_stale"} == first.comparison.identity,
                 "C2 diagnostic identity differs from C1")
        _require(later.comparable or "baseline.anchor_stale" in later.identity, "C2 is not comparable")
        _require(later.behavior == first.comparison.behavior
                 or (not later.behavior and "baseline.anchor_stale" in later.identity), "C2 behavior differs from C1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    _ = mode.add_argument("--check", type=Path)
    _ = mode.add_argument("--check-promotion", type=Path, nargs=2)
    _ = parser.add_argument("--bootstrap", action="store_true")
    _ = parser.add_argument("--accept-behavior-drift", action="store_true")
    arguments = parser.parse_args()
    try:
        record_path: Path | None = arguments.check
        pair: list[Path] | None = arguments.check_promotion
        bootstrap: bool = arguments.bootstrap
        accept_drift: bool = arguments.accept_behavior_drift
        _require(pair is not None or not (bootstrap or accept_drift), "promotion switches require --check-promotion")
        _require(not (bootstrap and accept_drift), "bootstrap and behavior acceptance are incompatible")
        if pair is not None:
            decision = Decision.BOOTSTRAP if bootstrap else Decision.ACCEPTED_BEHAVIOR if accept_drift else Decision.ROUTINE
            check_promotion((pair[0], pair[1]), decision=decision)
        elif record_path is not None:
            check_current(load_record(record_path, baseline=True))
        else:
            print(fingerprint())
    except (OSError, ValueError, RecursionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
