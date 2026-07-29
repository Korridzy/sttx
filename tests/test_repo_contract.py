from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

from repo_snapshot import canonical_snapshot


REPOSITORY = Path(__file__).parents[1]
ABSTRACTOR = Path("abstractor")
SOURCE_SPEC = ABSTRACTOR / ".omo/specs/deep-interview-sttx-python-transcriber.md"
COPIED_SPEC = REPOSITORY / "doc/specs/deep-interview-sttx.md"
AUTHOR = ("Korridzy", "korridzy@yandex.ru")
SUBJECTS = {
    "chore(sttx): establish the standalone project boundary",
    "feat(output): make transcript artifacts deterministic",
    "feat(audio): normalize media through a safe temporary wav",
    "feat(model): resolve a floating offline-ready asset bundle",
    "test(compat): gate the current sherpa and model surface",
    "feat(asr): assemble timestamped sentences from vad chunks",
    "feat(cli): expose the single-shot transcription command",
    "fix(cli): guarantee signal cleanup and conventional exits",
    "test(cli): lock offline media and error behavior",
    "test(integration): prove the floating local transcription pipeline",
    "docs(sttx): document installation usage and attribution",
    "build(sttx): prove the distributable command across supported Python",
    "chore(sttx): enforce the local-only repository contract",
}
CORRECTIVE_SUBJECT = re.compile(r"fix\([a-z]+\): close task (?:[1-9]|1[0-3]) verification gap\Z")
TRAILER_ORDER = (
    "Constraint",
    "Rejected",
    "Confidence",
    "Scope-risk",
    "Directive",
    "Tested",
    "Not-tested",
)
REQUIRED_TRAILERS = frozenset({"Constraint", "Confidence", "Scope-risk", "Tested", "Not-tested"})
OPTIONAL_TRAILERS = frozenset({"Rejected", "Directive"})
REQUIRED_FILES = frozenset(
    {
        ".gitignore",
        "README.md",
        "NOTICE.md",
        "poetry.toml",
        "pyproject.toml",
        "doc/specs/deep-interview-sttx.md",
        "src/sttx/__init__.py",
        "src/sttx/asr.py",
        "src/sttx/audio.py",
        "src/sttx/cli.py",
        "src/sttx/model.py",
        "src/sttx/output.py",
        "tests/conftest.py",
        "tests/repo_snapshot.py",
        "tests/test_packaging.py",
    }
)
FORBIDDEN_PREFIXES = (".github/", "deploy/", "deployment/", "infra/", "models/", "evidence/")
FORBIDDEN_NAMES = frozenset(
    {
        "poetry.lock",
        "Dockerfile",
        "docker-compose.yml",
        "compose.yml",
        "config.py",
        "settings.py",
    }
)
FORBIDDEN_SUFFIXES = (".onnx", ".whl", ".mp3", ".mp4", ".ogg", ".wav", ".xml")


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    return completed.stdout


def tracked_files(repository: Path) -> frozenset[str]:
    return frozenset(filter(None, git(repository, "ls-files").splitlines()))


def assert_history_contract(repository: Path) -> None:
    records = git(repository, "log", "--format=%H%x1f%an%x1f%ae%x1f%B%x1e").split("\x1e")
    defects: list[str] = []
    for record in filter(str.strip, records):
        commit, author_name, author_email, message = record.lstrip().split("\x1f", maxsplit=3)
        lines = message.strip().splitlines()
        subject = lines[0] if lines else ""
        if (author_name, author_email) != AUTHOR:
            defects.append(f"{commit}: author is {author_name} <{author_email}>")
        if subject not in SUBJECTS and CORRECTIVE_SUBJECT.fullmatch(subject) is None:
            defects.append(f"{commit}: invalid subject {subject!r}")
        if re.search(r"\bWIP\b", message, re.IGNORECASE):
            defects.append(f"{commit}: WIP marker")
        trailer_matches = re.findall(r"^([A-Za-z-]+):\s*(.+)$", message, re.MULTILINE)
        trailer_keys = [key for key, _ in trailer_matches]
        if any(key not in TRAILER_ORDER for key in trailer_keys):
            defects.append(f"{commit}: unknown decision trailer")
        if any(trailer_keys.count(key) != 1 for key in REQUIRED_TRAILERS):
            defects.append(f"{commit}: required trailer count")
        if any(trailer_keys.count(key) > 1 for key in OPTIONAL_TRAILERS):
            defects.append(f"{commit}: optional trailer count")
        if [key for key in trailer_keys if key in TRAILER_ORDER] != [
            key for key in TRAILER_ORDER if key in trailer_keys
        ]:
            defects.append(f"{commit}: trailer order")
        if re.search(r"^(?:AI|Co-Authored-By):", message, re.MULTILINE | re.IGNORECASE):
            defects.append(f"{commit}: prohibited trailer")
    assert not defects, "\n".join(defects)


def assert_repository_contract(repository: Path) -> None:
    assert git(repository, "branch", "--show-current").strip() == "main"
    assert not git(repository, "remote", "-v").strip()
    assert git(repository, "config", "--local", "user.name").strip() == AUTHOR[0]
    assert git(repository, "config", "--local", "user.email").strip() == AUTHOR[1]
    allowed_ordinary = {" M .gitignore", "?? tests/test_repo_contract.py"}
    status_lines = set(filter(None, git(repository, "status", "--short", "--ignored").splitlines()))
    ordinary_status = {line for line in status_lines if not line.startswith("!! ")}
    assert ordinary_status <= allowed_ordinary
    git(repository, "check-ignore", "--quiet", ".serena")
    assert all(line.startswith("!! ") or line in allowed_ordinary for line in status_lines)


def test_abstractor_matches_the_immutable_baseline(request: pytest.FixtureRequest) -> None:
    baseline = request.config.getoption("--abstractor-baseline")
    expected_digest = request.config.getoption("--abstractor-baseline-sha256")
    assert isinstance(baseline, Path), "--abstractor-baseline is required"
    assert isinstance(expected_digest, str) and expected_digest, "--abstractor-baseline-sha256 is required"
    assert baseline == Path("task-1-abstractor-baseline.json")
    assert expected_digest == "7a3776ee50a165ca3c203093cab2c45c256e678c9c87262efd8f01f4399f45b9"
    payload = baseline.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_digest
    assert baseline.stat().st_mode & 0o777 == 0o444
    assert canonical_snapshot(ABSTRACTOR) == payload


def test_repository_layout_matches_the_approved_boundary() -> None:
    assert SOURCE_SPEC.read_bytes() == COPIED_SPEC.read_bytes()
    tracked = tracked_files(REPOSITORY)
    assert REQUIRED_FILES <= tracked
    forbidden = sorted(
        path
        for path in tracked
        if path in FORBIDDEN_NAMES
        or path.startswith(FORBIDDEN_PREFIXES)
        or path.endswith(FORBIDDEN_SUFFIXES)
        or "/__pycache__/" in path
        or path.startswith(".venv/")
        or path.startswith("dist/")
        or path.startswith(".pytest_cache/")
        or ".egg-info/" in path
    )
    assert not forbidden, forbidden


def test_repository_state_matches_the_local_only_contract() -> None:
    assert_repository_contract(REPOSITORY)


def test_repository_history_matches_the_decision_record_contract() -> None:
    assert_history_contract(REPOSITORY)


def test_repository_contract_rejects_a_remote(tmp_path: Path) -> None:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", AUTHOR[0])
    git(tmp_path, "config", "user.email", AUTHOR[1])
    git(tmp_path, "remote", "add", "origin", "https://example.invalid/sttx.git")
    with pytest.raises(AssertionError):
        assert_repository_contract(tmp_path)
