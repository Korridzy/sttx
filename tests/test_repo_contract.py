from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).parents[1]
AUTHOR = ("Korridzy", "Korridzy@yandex.ru")
REQUIRED_FILES = frozenset(
    {
        ".gitignore",
        "LICENSE",
        "README.md",
        "NOTICE.md",
        "poetry.toml",
        "pyproject.toml",
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
FORBIDDEN_PREFIXES = (
    ".github/",
    "deploy/",
    "deployment/",
    "doc/",
    "infra/",
    "models/",
    "evidence/",
)
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


def test_repository_layout_matches_the_approved_boundary() -> None:
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


def test_repository_contract_rejects_a_remote(tmp_path: Path) -> None:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", AUTHOR[0])
    git(tmp_path, "config", "user.email", AUTHOR[1])
    git(tmp_path, "remote", "add", "origin", "https://example.invalid/sttx.git")
    with pytest.raises(AssertionError):
        assert_repository_contract(tmp_path)
