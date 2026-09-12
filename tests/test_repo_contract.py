from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Final

import pytest

BACKEND_FILES: Final = frozenset(
    {
        ".github/dependabot.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        "KNOWN-GAPS.md",
        "src/sttx/backend_contract.py",
        "scripts/compute_backend_fingerprint.py",
        "scripts/backend_change_detector.py",
        "tests/backend_qualification_helpers.py",
        "tests/test_backend_contract.py",
        "tests/test_backend_fingerprint.py",
        "tests/test_backend_promotion.py",
        "tests/test_backend_change_detector.py",
        "tests/test_backend_workflows.py",
        "tests/integration/timing_lattice.py",
        "tests/integration/test_timing_lattice_unit.py",
        "tests/integration/test_timing_qualification.py",
        "tests/integration/qualification_schema.py",
        "tests/integration/qualification_validation.py",
        "tests/integration/qualification_evaluation.py",
        "tests/integration/qualification_recording.py",
        "tests/integration/qualification_probes.py",
        "tests/integration/test_qualification_schema.py",
        "tests/integration/test_qualification_evaluation.py",
        "tests/integration/test_qualification_recording.py",
        "tests/integration/test_qualification_lifecycle.py",
        "tests/integration/qualified_backend.json",
    }
)

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
        ".github/dependabot.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        "KNOWN-GAPS.md",
        "src/sttx/backend_contract.py",
        "scripts/compute_backend_fingerprint.py",
        "scripts/backend_change_detector.py",
        "tests/backend_qualification_helpers.py",
        "tests/test_backend_contract.py",
        "tests/test_backend_fingerprint.py",
        "tests/test_backend_promotion.py",
        "tests/test_backend_change_detector.py",
        "tests/test_backend_workflows.py",
        "tests/integration/timing_lattice.py",
        "tests/integration/test_timing_lattice_unit.py",
        "tests/integration/test_timing_qualification.py",
        "tests/integration/qualification_schema.py",
        "tests/integration/qualification_validation.py",
        "tests/integration/qualification_evaluation.py",
        "tests/integration/qualification_recording.py",
        "tests/integration/qualification_probes.py",
        "tests/integration/test_qualification_schema.py",
        "tests/integration/test_qualification_evaluation.py",
        "tests/integration/test_qualification_recording.py",
        "tests/integration/test_qualification_lifecycle.py",
        "tests/integration/qualified_backend.json",
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
    assert REQUIRED_FILES <= tracked, sorted(REQUIRED_FILES - tracked)
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


def test_required_inventory_when_backend_policy_is_shipped() -> None:
    required = REQUIRED_FILES

    assert BACKEND_FILES <= required


@pytest.mark.parametrize("omitted", [None, *sorted(BACKEND_FILES)])
def test_layout_when_backend_path_is_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, omitted: str | None,
) -> None:
    _ = git(tmp_path, "init", "--quiet")
    for name in REQUIRED_FILES | BACKEND_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    included = sorted((REQUIRED_FILES | BACKEND_FILES) - {omitted})
    _ = git(tmp_path, "add", "--", *included)
    monkeypatch.setattr(f"{__name__}.REPOSITORY", tmp_path)

    if omitted is None:
        test_repository_layout_matches_the_approved_boundary()
    else:
        with pytest.raises(AssertionError, match=re.escape(omitted)):
            test_repository_layout_matches_the_approved_boundary()
