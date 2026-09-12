"""Read-only semantic backend gate, runnable with Python 3.11+ -I -S -B.

Checkout --head before invoking; fetch comparison objects externally. Only the
current checkout's trusted fingerprint checker/contract are loaded. Git blobs
are data, never executable modules. Missing base explicitly means introduction.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol

ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT: Final = "src/sttx/backend_contract.py"
RECORD: Final = "tests/integration/qualified_backend.json"
JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class DetectorError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Snapshot:
    contract_text: str | None
    record_present: bool


@dataclass(frozen=True, slots=True)
class Changes:
    changed_paths: frozenset[str]
    fingerprint_matches: bool
    bound_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    needs_qualification: bool
    fingerprint_mismatch: bool
    record_changed: bool
    changed_sources: tuple[str, ...]
    first_introduction: bool
    base_version: int | None
    head_version: int


def probe_version(text: str) -> int:
    """Accept one positive top-level integer literal, without executing text."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise DetectorError("malformed contract AST") from error
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
              and node.id == "PROBE_VERSION" and isinstance(node.ctx, (ast.Store, ast.Del))]
    rebound = any(
        getattr(node, "name", None) == "PROBE_VERSION"
        or (isinstance(node, ast.MatchMapping) and node.rest == "PROBE_VERSION")
        or (isinstance(node, ast.arg) and node.arg == "PROBE_VERSION")
        or (isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) in {"PROBE_VERSION", "*"})
        for node in ast.walk(tree))
    values: list[ast.expr | None] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "PROBE_VERSION":
                values.append(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PROBE_VERSION":
                values.append(node.value)
    if rebound or len(writes) != 1 or len(values) != 1:
        raise DetectorError("PROBE_VERSION requires exactly one literal assignment")
    value = values[0]
    if not isinstance(value, ast.Constant) or type(value.value) is not int or value.value < 1:
        raise DetectorError("PROBE_VERSION must be a positive integer literal, not bool")
    return value.value


def decide(base: Snapshot, head: Snapshot, changes: Changes) -> Decision:
    """Pure policy over supplied snapshots, semantic fingerprint and path diff."""
    if head.contract_text is None or not head.record_present:
        raise DetectorError("head requires both contract and qualification record")
    if (base.contract_text is not None) != base.record_present:
        raise DetectorError("base has one-sided contract/record absence")
    head_version = probe_version(head.contract_text)
    base_version = None if base.contract_text is None else probe_version(base.contract_text)
    first = base_version is None
    record_changed = RECORD in changes.changed_paths
    sources = tuple(sorted(set(changes.bound_sources) & changes.changed_paths))
    if first and (head_version != 1 or not record_changed):
        raise DetectorError("first introduction requires version 1 and a new record")
    if base_version is not None:
        if head_version < base_version:
            raise DetectorError("PROBE_VERSION cannot decrease")
        if sources and (not record_changed or head_version <= base_version):
            raise DetectorError("changed bound sources require changed record and increased PROBE_VERSION")
    return Decision(not changes.fingerprint_matches or record_changed,
                    not changes.fingerprint_matches, record_changed, sources,
                    first, base_version, head_version)


@dataclass(frozen=True, slots=True)
class Comparison:
    base: Snapshot
    head: Snapshot
    changed_paths: frozenset[str]
    head_commit: str
    base_missing: bool
    record_text: str


def _git(root: Path, args: tuple[str, ...]) -> str:
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
        capture_output=True, text=True, encoding="utf-8", errors="surrogateescape", timeout=30,
        env={**os.environ, "GIT_MASTER": "1", "GIT_NO_LAZY_FETCH": "1"})
    if result.returncode:
        raise DetectorError(result.stderr.strip() or "git read failed")
    return result.stdout


def read_comparison(root: Path, base: str, head: str) -> Comparison:
    head_commit = _git(root, ("rev-parse", "--verify", "--end-of-options", head + "^{commit}")).strip()
    base_commit: str | None = None
    if base and set(base) != {"0"}:
        try:
            base_commit = _git(root, ("rev-parse", "--verify", "--end-of-options", base + "^{commit}")).strip()
        except DetectorError:
            base_commit = None
    head_paths = frozenset(_git(root, ("ls-tree", "-r", "--name-only", "-z", head_commit)).split("\0")) - {""}
    base_paths: frozenset[str] = (frozenset(_git(root, ("ls-tree", "-r", "--name-only", "-z", base_commit)).split("\0"))
                  if base_commit is not None else frozenset())
    changed = (frozenset(_git(root, ("diff", "--no-ext-diff", "--no-renames", "--name-only", "-z",
                                   base_commit, head_commit, "--")).split("\0")) - {""}
               if base_commit is not None else head_paths)
    base_text = (_git(root, ("show", base_commit + ":" + CONTRACT))
                 if base_commit is not None and CONTRACT in base_paths else None)
    head_text = _git(root, ("show", head_commit + ":" + CONTRACT)) if CONTRACT in head_paths else None
    record_text = _git(root, ("show", head_commit + ":" + RECORD)) if RECORD in head_paths else ""
    return Comparison(Snapshot(base_text, RECORD in base_paths), Snapshot(head_text, RECORD in head_paths),
                      changed, head_commit, base_commit is None, record_text)


def _checker() -> ModuleType:
    path = ROOT / "scripts/compute_backend_fingerprint.py"
    spec = importlib.util.spec_from_file_location("_sttx_detector_checker", path)
    if spec is None or spec.loader is None:
        raise DetectorError("cannot load trusted current checker")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    bytecode = sys.dont_write_bytecode
    sys.modules[spec.name] = module
    sys.dont_write_bytecode = True
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    finally:
        sys.dont_write_bytecode = bytecode
        if previous is None:
            del sys.modules[spec.name]
        else:
            sys.modules[spec.name] = previous
    return module


class Fingerprint(Protocol):
    def __call__(self, root: Path, *, require_sources: bool = False) -> str: ...


def _unique_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise DetectorError(f"duplicate record key: {key}")
        result[key] = value
    return result


def current_fingerprint(root: Path, head_commit: str, sources: tuple[str, ...]) -> str:
    """Reject mixed worktrees before executing the trusted current contract."""
    for name in (CONTRACT, RECORD, "pyproject.toml", *sources):
        committed = _git(root, ("show", head_commit + ":" + name)).encode("utf-8", "surrogateescape")
        if (root / name).is_symlink() or (root / name).read_bytes() != committed:
            raise DetectorError(f"checkout differs from --head: {name}")
    if root.resolve() != ROOT:
        raise DetectorError("--root must be the checkout containing this detector")
    fingerprint: Fingerprint = _checker().fingerprint
    return fingerprint(root, require_sources=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--base", default="", help="base revision; empty/all-zero/unavailable means introduction")
    _ = parser.add_argument("--head", default="HEAD", help="checked-out head revision (default HEAD)")
    _ = parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    root: Path = arguments.root
    base: str = arguments.base
    head: str = arguments.head
    try:
        comparison = read_comparison(root, base, head)
        sources: tuple[str, ...] = _checker().SOURCES
        preliminary = Changes(comparison.changed_paths, False, sources)
        _ = decide(comparison.base, comparison.head, preliminary)
        record: JsonValue = json.loads(comparison.record_text, object_pairs_hook=_unique_pairs)
        if not isinstance(record, dict):
            raise DetectorError("record requires a fingerprint string")
        recorded = record.get("fingerprint")
        if not isinstance(recorded, str) or re.fullmatch("[0-9a-f]{64}", recorded) is None:
            raise DetectorError("record fingerprint must be a SHA-256")
        fingerprint = current_fingerprint(root, comparison.head_commit, sources)
        decision = decide(comparison.base, comparison.head,
                          Changes(comparison.changed_paths, recorded == fingerprint, sources))
        print(json.dumps({**asdict(decision), "base_missing": comparison.base_missing,
                          "head_commit": comparison.head_commit}, sort_keys=True))
    except (DetectorError, OSError, ValueError, RecursionError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
