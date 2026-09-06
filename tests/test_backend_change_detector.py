from __future__ import annotations

import json
import importlib.util
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import scripts.backend_change_detector as detector
from scripts import compute_backend_fingerprint as checker

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/backend_change_detector.py"
BINDINGS = ("match 99:\n case PROBE_VERSION: pass", "match []:\n case [*PROBE_VERSION]: pass",
    "match {}:\n case {**PROBE_VERSION}: pass", "try: pass\nexcept Exception as PROBE_VERSION: pass",
    "def method(PROBE_VERSION): pass", "import PROBE_VERSION.child", "from other import *")


@pytest.mark.parametrize("text", ["PROBE_VERSION = 1", "PROBE_VERSION: Final = 12"])
def test_version_when_integer_literal_returns_value(text: str) -> None:
    expected = 12 if "12" in text else 1

    result = detector.probe_version(text)

    assert result == expected


@pytest.mark.parametrize("text", ["", "PROBE_VERSION =", "PROBE_VERSION = True",
    "PROBE_VERSION = 1.0", "PROBE_VERSION = '1'", "PROBE_VERSION = 1 + 1",
    "PROBE_VERSION = -1", "PROBE_VERSION = 0", "PROBE_VERSION: int",
    "PROBE_VERSION = 1\nPROBE_VERSION = 2", "PROBE_VERSION = other = 1",
    "PROBE_VERSION = 1\nif False:\n PROBE_VERSION = 2",
    "PROBE_VERSION = 1\nPROBE_VERSION += 1", "PROBE_VERSION, other = (1, 2)"])
def test_version_when_malformed_rejects(text: str) -> None:
    with pytest.raises(detector.DetectorError):
        _ = detector.probe_version(text)


@pytest.mark.parametrize("paths,matches,expected", [
    (frozenset({"README.md", "pyproject.toml"}), True, False),
    (frozenset({"pyproject.toml"}), False, True),
    (frozenset({detector.RECORD}), True, True),
    (frozenset({detector.CONTRACT}), False, True),
])
def test_decision_when_semantic_trigger_controls_qualification(
        paths: frozenset[str], matches: bool, expected: bool) -> None:
    snapshot = detector.Snapshot("PROBE_VERSION = 2", True)
    changes = detector.Changes(paths, matches, checker.SOURCES)

    result = detector.decide(snapshot, snapshot, changes)

    assert result.needs_qualification is expected


@pytest.mark.parametrize("source", checker.SOURCES)
@pytest.mark.parametrize("version,record,allowed", [(3, True, True), (2, True, False),
    (1, True, False), (3, False, False)])
def test_bound_source_when_changed_requires_record_and_version(
        source: str, version: int, record: bool, allowed: bool) -> None:
    paths = frozenset({source, detector.RECORD} if record else {source})
    changes = detector.Changes(paths, True, checker.SOURCES)
    base = detector.Snapshot("PROBE_VERSION = 2", True)
    head = detector.Snapshot(f"PROBE_VERSION = {version}", True)

    if allowed:
        result = detector.decide(base, head, changes)
        assert result.needs_qualification and result.changed_sources == (source,)
    else:
        with pytest.raises(detector.DetectorError):
            _ = detector.decide(base, head, changes)


@pytest.mark.parametrize("base_text,base_record,head_text,head_record,allowed", [
    (None, False, "PROBE_VERSION = 1", True, True),
    (None, False, "PROBE_VERSION = True", True, False),
    (None, False, "PROBE_VERSION = 2", True, False),
    (None, False, None, False, False),
    (None, False, "PROBE_VERSION = 1", False, False),
    (None, True, "PROBE_VERSION = 1", True, False),
    ("PROBE_VERSION = 1", False, "PROBE_VERSION = 1", True, False),
    ("PROBE_VERSION = 1", True, None, True, False),
    ("PROBE_VERSION = 1", True, "PROBE_VERSION = 1", False, False),
])
def test_introduction_when_presence_is_checked(base_text: str | None, base_record: bool,
        head_text: str | None, head_record: bool, allowed: bool) -> None:
    base = detector.Snapshot(base_text, base_record)
    head = detector.Snapshot(head_text, head_record)
    changes = detector.Changes(frozenset({detector.RECORD, *checker.SOURCES}), True, checker.SOURCES)

    if allowed:
        result = detector.decide(base, head, changes)
        assert result.first_introduction and result.needs_qualification
    else:
        with pytest.raises(detector.DetectorError):
            _ = detector.decide(base, head, changes)


def test_ast_when_historical_statements_are_poisoned_never_executes(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    text = f"PROBE_VERSION = 1\nopen({str(marker)!r}, 'w').close()\nraise RuntimeError()"

    result = detector.probe_version(text)

    assert result == 1 and not marker.exists()


@pytest.mark.parametrize("base", ["", "0" * 40, "missing"])
def test_git_when_base_unavailable_takes_explicit_introduction(
        monkeypatch: pytest.MonkeyPatch, base: str) -> None:
    def git(_root: Path, args: tuple[str, ...]) -> str:
        if args[0] == "rev-parse":
            if args[-1].startswith("missing"):
                raise detector.DetectorError("missing object")
            return "a" * 40
        if args[0] == "ls-tree":
            return detector.CONTRACT + "\0" + detector.RECORD + "\0"
        assert args[0] == "show"
        return "PROBE_VERSION = 1" if args[-1].endswith(detector.CONTRACT) else '{}'
    monkeypatch.setattr(detector, "_git", git)

    comparison = detector.read_comparison(ROOT, base, "HEAD")

    assert comparison.base == detector.Snapshot(None, False)
    assert comparison.base_missing
    assert detector.RECORD in comparison.changed_paths


@pytest.mark.parametrize("mode", ["metadata", "dependency", "record", "probe", "version", "malformed",
    "duplicate", "boolean", "digest"])
def test_cli_when_git_is_injected_runs_isolated_without_path_pollution(mode: str) -> None:
    driver = (
        "import runpy,sys,json\nbefore=list(sys.path)\n"
        f"ns=runpy.run_path({str(SCRIPT)!r})\nscope=ns['main'].__globals__\nmode={mode!r}\n"
        "def git(root,args):\n"
        " if args[0]=='rev-parse': return ('b' if args[-1].startswith('base') else 'a')*40\n"
        " if args[0]=='ls-tree': return ns['CONTRACT']+'\\0'+ns['RECORD']+'\\0'\n"
        " if args[0]=='diff':\n"
        "  paths=['pyproject.toml']\n"
        "  if mode in ('record','probe','version'): paths.append(ns['RECORD'])\n"
        "  if mode in ('probe','version'): paths.append(scope['_checker']().SOURCES[-1])\n"
        "  return '\\0'.join(paths)+'\\0'\n"
        " assert args[0]=='show'\n"
        " if args[-1].endswith(ns['CONTRACT']):\n"
        "  version=2 if mode=='probe' and args[-1].startswith('a') else 1\n"
        "  return f'PROBE_VERSION = {version}\\nraise RuntimeError(\"never execute Git\")'\n"
        " if mode=='duplicate': return '{\"fingerprint\":\"wrong\",\"fingerprint\":\"'+'f'*64+'\"}'\n"
        " if mode=='boolean': return json.dumps({'fingerprint':True})\n"
        " if mode=='digest': return json.dumps({'fingerprint':'invalid'})\n"
        " return '[]' if mode=='malformed' else json.dumps({'fingerprint':'f'*64})\n"
        "scope['_git']=git\nscope['current_fingerprint']=lambda *args: ('e' if mode=='dependency' else 'f')*64\n"
        "status=ns['main'](['--base','base'])\nassert sys.path==before\n"
        "assert not any(n.split('.')[0] in {'sttx','numpy','sherpa_onnx','pytest'} for n in sys.modules)\n"
        "raise SystemExit(status)"
    )

    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", driver],
                            capture_output=True, text=True, timeout=10)

    if mode in {"version", "malformed", "duplicate", "boolean", "digest"}:
        assert result.returncode == 1 and result.stdout == ""
        assert result.stderr.startswith("error:") and "Traceback" not in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["needs_qualification"] is (mode != "metadata")
        assert result.stderr == ""


def test_cli_when_head_is_invalid_errors_without_writes() -> None:
    result = subprocess.run([sys.executable, "-I", "-S", "-B", str(SCRIPT),
        "--base", "0" * 40, "--head", "not-a-real-revision"],
        capture_output=True, text=True, timeout=10)

    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr.startswith("error:") and "Traceback" not in result.stderr


def test_git_when_real_head_compares_to_itself_is_unchanged() -> None:
    result = detector.read_comparison(ROOT, "HEAD", "HEAD")

    assert result.changed_paths == frozenset() and not result.base_missing
    assert result.base == result.head and result.head.contract_text is not None


@pytest.mark.parametrize("suffix", ["\nimport other as PROBE_VERSION",
    "\ndef PROBE_VERSION(): pass", "\nclass PROBE_VERSION: pass", *("\n" + text for text in BINDINGS)])
def test_version_when_binding_is_overwritten_rejects(suffix: str) -> None:
    with pytest.raises(detector.DetectorError):
        _ = detector.probe_version("PROBE_VERSION = 1" + suffix)


@pytest.mark.parametrize("changed", [False, True])
def test_current_fingerprint_when_git_bytes_match_uses_canonical_checker(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: bool) -> None:
    for name in (detector.CONTRACT, detector.RECORD, "pyproject.toml", *checker.SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes((ROOT / name).read_bytes() if (ROOT / name).exists()
                             else b"raise RuntimeError('hashed only, never executed')\n")
    monkeypatch.setattr(detector, "ROOT", tmp_path)
    def git(_root: Path, args: tuple[str, ...]) -> str:
        assert args[0] == "show"
        name = args[-1].split(":", 1)[1]
        return (tmp_path / name).read_text() + ("\n" if changed else "")
    monkeypatch.setattr(detector, "_git", git)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    if changed:
        with pytest.raises(detector.DetectorError):
            _ = detector.current_fingerprint(tmp_path, "a" * 40, checker.SOURCES)
    else:
        result = detector.current_fingerprint(tmp_path, "a" * 40, checker.SOURCES)
        assert result == checker.fingerprint(tmp_path, require_sources=True)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def git(root: Path, args: tuple[str, ...], data: bytes | None = None) -> str:
    env = {**os.environ, "GIT_MASTER": "1", "GIT_AUTHOR_NAME": "Test", "GIT_COMMITTER_NAME": "Test",
           "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
    return subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True,
                          env=env, timeout=15, check=True).stdout.decode().strip()


def tree(root: Path, files: dict[str, bytes]) -> str:
    entries: list[str] = []
    for folder in sorted({name.split("/", 1)[0] for name in files if "/" in name}):
        nested = {name.split("/", 1)[1]: data for name, data in files.items() if name.startswith(folder + "/")}
        entries.append(f"040000 tree {tree(root, nested)}\t{folder}\n")
    for name, data in sorted(files.items()):
        if "/" not in name:
            blob = git(root, ("hash-object", "-w", "--stdin"), data)
            entries.append(f"100644 blob {blob}\t{name}\n")
    return git(root, ("mktree",), "".join(entries).encode())


@pytest.mark.parametrize("mode", ["pyc", "dirty", "deleted", "symlink", "untracked", *BINDINGS])
def test_actual_git_cli_when_untrusted_state_cannot_fast_pass(mode: str) -> None:
    with tempfile.TemporaryDirectory(prefix="detector-regression-") as temporary:
        root = Path(temporary)
        files = {name: (ROOT / name).read_bytes() for name in (*checker.SOURCES, detector.CONTRACT, "pyproject.toml")}
        for name, data in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            _ = (root / name).write_bytes(data)
        digest = checker.fingerprint(root, require_sources=True)
        files[detector.RECORD] = json.dumps({"fingerprint": digest}).encode()
        _ = (root / detector.RECORD).write_bytes(files[detector.RECORD])
        _ = git(root, ("init", "-q"))
        base_files = files | ({detector.CONTRACT: ("PROBE_VERSION = 1\n" + mode).encode()} if mode in BINDINGS else {})
        base = git(root, ("commit-tree", tree(root, base_files)), b"base\n")
        if mode == "pyc":
            files["pyproject.toml"] = files["pyproject.toml"].replace(b"numpy==2.4.6", b"numpy==2.4.7")
            _ = (root / "pyproject.toml").write_bytes(files["pyproject.toml"])
            source = root / "cached.py"
            _ = source.write_text(f"SOURCES={checker.SOURCES!r}\ndef fingerprint(*args, **kwargs): return {digest!r}\n")
            cached = Path(importlib.util.cache_from_source(str(root / "scripts/compute_backend_fingerprint.py")))
            cached.parent.mkdir(exist_ok=True)
            _ = py_compile.compile(str(source), cfile=str(cached), doraise=True,
                                   invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
        head = git(root, ("commit-tree", tree(root, files)), b"head\n")
        if mode == "dirty":
            _ = (root / detector.RECORD).write_text("malformed dirty record")
        if mode in {"deleted", "symlink"}:
            (root / detector.RECORD).unlink()
        if mode == "symlink":
            _ = (root / "record-copy").write_bytes(files[detector.RECORD])
            (root / detector.RECORD).symlink_to(root / "record-copy")
        _ = (root / "uv.lock").write_text("unrelated untracked state")
        before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(root / "scripts/backend_change_detector.py"),
            "--base", base, "--head", head], capture_output=True, text=True, timeout=15)

        if mode in {"pyc", "untracked"}:
            assert result.returncode == 0 and result.stderr == ""
            assert json.loads(result.stdout)["needs_qualification"] is (mode == "pyc")
        else:
            assert result.returncode == 1 and result.stdout == ""
            assert result.stderr.startswith("error:") and "Traceback" not in result.stderr
        assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
