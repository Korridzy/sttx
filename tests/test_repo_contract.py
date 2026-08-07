from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).parents[1]
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
