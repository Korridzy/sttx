"""Read-only qualification identity. Print permits an absent future gate only.

Run with Python 3.11+ (including -I -S); no installed distribution is required.
Release checks require all six sources. Only this checkout's trusted contract
and pure lattice helper are executed; historical/base modules are never loaded.
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
    from tests.integration.timing_lattice import JsonValue, RawObservation, SignedPayload

ROOT: Final = Path(__file__).resolve().parents[1]
SOURCES: Final = ("scripts/compute_backend_fingerprint.py", "tests/integration/timing_lattice.py",
    "tests/integration/test_timing_qualification.py", "tests/conftest.py",
    "src/sttx/asr.py", "src/sttx/cli.py")


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_sttx_qualification_" + path.stem, path)
    if spec is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
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

STALE: Final = frozenset("baseline." + name + "_stale" for name in (
    "fingerprint", "probe_version", "model_repo", "model_revision", "model_assets", "dependencies", "quantum"))
BEHAVIOR: Final = frozenset({"behavior.length", "behavior.identity", "behavior.timing"})
COLLECTIONS: Final = frozenset({"runs", "production_runs", "production_transcripts"})
RECORD_KEYS: Final = COLLECTIONS | {
    "record_kind", "schema_version", "fingerprint", "probe_version", "model", "dependencies", "fixture",
    "platform", "derived_quantum_us", "inconclusive_reason", "observed_max_overhang_us",
    "signature_sha256", "baseline_comparison"}


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
    decoded: JsonValue = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=lattice._unique_pairs)
    data = lattice._shape(decoded, RECORD_KEYS | ({"promotion"} if baseline else set()))
    _require(data["record_kind"] == ("qualified_backend" if baseline else "qualification_candidate"),
             "incorrect record kind")
    contract = contract_payload(root)
    expected = {
        "model": {"repo": contract["parakeet_repo_id"], "revision": contract["parakeet_revision"],
                  "sha256": {**contract["parakeet_sha256"], contract["silero_filename"]: contract["silero_sha256"]}},
        "fixture": {"repo": contract["fixture_repo_id"], "revision": contract["fixture_revision"],
                    "name": contract["fixture_filename"], "sha256": contract["fixture_sha256"]},
        "dependencies": dict(item.split("==") for item in dependencies(root)),
        "platform": {"system": platform.system(), "machine": platform.machine()},
    }
    _require(platform.system() == "Linux", "unsupported qualification platform")
    for name, identity in expected.items():
        _require(data[name] == identity, f"candidate {name} differs from current contract")
    _require(lattice._integer(data["schema_version"]) == 1, "unsupported schema version")
    _require(lattice._integer(data["probe_version"]) == contract["probe_version"], "candidate probe version")
    payload = lattice.load_payload(json.dumps({name: data[name] for name in COLLECTIONS}))
    signature = lattice.signature_sha256(payload)
    _require(_digest(data["signature_sha256"]) == signature, "signed payload mismatch")
    quantum = lattice.derive_quantum_us([time for run in payload["runs"] for time in run["timestamps_us"]])
    _require(lattice._integer(data["derived_quantum_us"]) == quantum == contract["encoder_frame_us"]
             and data["inconclusive_reason"] is None, "invalid or inconclusive candidate quantum")
    transcripts = payload["production_transcripts"]
    _require([item["probe"] for item in transcripts] == ["original", "long"], "missing production probes")
    _require(transcripts[1]["duration_us"] == 36_000_000, "long probe must contain 576000 samples")
    _require(all(item["text"].strip() and item["segments"] for item in transcripts), "missing fixture speech")
    _require(all(item["text"] == " ".join(" ".join(segment["text"].split())
                 for segment in item["segments"] if segment["text"].strip()) for item in transcripts),
             "transcript text differs from normalized segments")
    _require(any(run["probe"] == "long" and run["chunk_index"] == 1 for run in payload["production_runs"]),
             "missing long multichunk coverage")
    overhang = 0
    try:
        observations: list[tuple[RawObservation, float]] = [
                        (run, transcripts[0]["duration_us"] + run["variant"] * 1_000_000 / 16_000)
                        for run in payload["runs"]]
        observations.extend((run, run["sample_count"] * 1_000_000 / 16_000) for run in payload["production_runs"])
        for run, duration in observations:
            _require(bool(run["tokens"]) and bool(run["text"].strip()), "missing raw speech coverage")
            ends = [start + (run["durations_us"][index] if run["durations_us"] else 0)
                    for index, start in enumerate(run["timestamps_us"])]
            overhang = max(overhang, lattice.quantize_us(max(0, max(ends) - duration) / 1_000_000))
    except OverflowError as error:
        raise lattice.ObservationError("observation arithmetic exceeds supported numeric magnitude") from error
    _require(lattice._integer(data["observed_max_overhang_us"]) == overhang <= contract["max_token_overhang_us"],
             "candidate overhang mismatch or excess")
    comparison = load_comparison(data["baseline_comparison"])
    if baseline:
        promotion = lattice._shape(data["promotion"], frozenset({
            "previous_fingerprint", "previous_anchor", "comparison_to_previous", "decision"}))
        _ = load_comparison(promotion["comparison_to_previous"])
        _require(promotion["comparison_to_previous"] == data["baseline_comparison"], "provenance comparison mismatch")
        decision = Decision(lattice._string(promotion["decision"]))
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
