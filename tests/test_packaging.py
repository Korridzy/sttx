from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WHEEL = "sttx-0.1.0-py3-none-any.whl"
DIST_INFO = "sttx-0.1.0.dist-info"
EXPECTED_SOURCES = {
    "sttx/__init__.py",
    "sttx/asr.py",
    "sttx/audio.py",
    "sttx/cli.py",
    "sttx/model.py",
    "sttx/output.py",
}
EXPECTED_METADATA = {
    f"{DIST_INFO}/METADATA",
    f"{DIST_INFO}/WHEEL",
    f"{DIST_INFO}/entry_points.txt",
    f"{DIST_INFO}/licenses/LICENSE",
    f"{DIST_INFO}/licenses/NOTICE.md",
    f"{DIST_INFO}/RECORD",
}
FORBIDDEN_WHEEL_PARTS = {
    ".git",
    ".omo",
    ".pytest_cache",
    ".serena",
    ".venv",
    "__pycache__",
    "dist",
    "poetry.lock",
    "transcriptions",
}
FORBIDDEN_ASSET_SUFFIXES = {
    ".bin",
    ".cache",
    ".ckpt",
    ".gguf",
    ".nemo",
    ".onnx",
    ".pt",
    ".safetensors",
}


def test_pep_621_metadata_defines_supported_console_distribution() -> None:
    # Given: the project metadata used by Poetry.
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    # When: the PEP 621 project table and scripts are inspected.
    project = pyproject["project"]
    scripts = project["scripts"]

    # Then: the supported distribution contract is explicit and version bounded.
    assert project["name"] == "sttx"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.11,<3.14"
    assert scripts == {"sttx": "sttx.cli:main"}


def test_poetry_lockfile_is_local_ignored_state() -> None:
    # Given: Poetry may create a local lockfile that must not be distributed.
    lockfile = "poetry.lock"

    # When: git classifies that path.
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", lockfile],
        cwd=PROJECT_ROOT,
        check=False,
        timeout=10,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "poetry.lock"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    # Then: the lockfile stays ignored local state, not a tracked artifact.
    assert ignored.returncode == 0
    assert tracked.returncode != 0


def test_shipped_sources_compile_with_supported_python_311() -> None:
    # Given: Python 3.11 is a declared supported interpreter.
    python_311 = shutil.which("python3.11")
    source_paths = sorted((PROJECT_ROOT / "src" / "sttx").glob("*.py"))

    # When: its parser compiles every shipped source module without importing it.
    completed = subprocess.run(
        [
            python_311 or "python3.11",
            "-c",
            (
                "from pathlib import Path; import sys; "
                "[compile(Path(path).read_text(), path, 'exec') for path in sys.argv[1:]]"
            ),
            *(os.fspath(path) for path in source_paths),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )

    # Then: package syntax is usable on the full declared Python range.
    assert python_311 is not None
    assert completed.returncode == 0, completed.stderr


def test_built_wheel_has_pure_tag_entry_point_and_exact_inventory(
    tmp_path: Path,
) -> None:
    # Given: a clean wheel output directory outside the repository.
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    # When: Poetry builds the distributable wheel.
    completed = subprocess.run(
        [
            "poetry",
            "build",
            "--format",
            "wheel",
            "--output",
            os.fspath(wheel_dir),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )

    # Then: the archive name, tag, entry point, metadata, and files are exact.
    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in wheel_dir.iterdir()) == [EXPECTED_WHEEL]
    wheel_path = wheel_dir / EXPECTED_WHEEL
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        wheel_metadata = Parser().parsestr(
            archive.read(f"{DIST_INFO}/WHEEL").decode(),
        )
        package_metadata = Parser().parsestr(
            archive.read(f"{DIST_INFO}/METADATA").decode(),
        )
        entry_points = configparser.ConfigParser()
        entry_points.read_string(
            archive.read(f"{DIST_INFO}/entry_points.txt").decode(),
        )

    assert names == EXPECTED_SOURCES | EXPECTED_METADATA
    assert wheel_metadata["Root-Is-Purelib"] == "true"
    assert wheel_metadata.get_all("Tag") == ["py3-none-any"]
    assert package_metadata["Name"] == "sttx"
    assert package_metadata["Version"] == "0.1.0"
    assert package_metadata["Requires-Python"] == ">=3.11,<3.14"
    assert package_metadata["License-Expression"] == "Apache-2.0"
    assert package_metadata.get_all("License-File") == ["LICENSE", "NOTICE.md"]
    assert sorted(package_metadata.get_all("Requires-Dist")) == [
        "huggingface-hub",
        "numpy",
        "sherpa-onnx",
        "sherpa-onnx-bin",
    ]
    assert entry_points["console_scripts"]["sttx"] == "sttx.cli:main"


def test_built_wheel_excludes_models_caches_and_local_artifacts(
    tmp_path: Path,
) -> None:
    # Given: a freshly built wheel.
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    subprocess.run(
        [
            "poetry",
            "build",
            "--format",
            "wheel",
            "--output",
            os.fspath(wheel_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )

    # When: its archive paths are inspected.
    with zipfile.ZipFile(wheel_dir / EXPECTED_WHEEL) as archive:
        names = archive.namelist()

    # Then: no model/cache assets or local runtime artifacts are bundled.
    assert all(
        not (FORBIDDEN_WHEEL_PARTS & set(Path(name).parts)) for name in names
    )
    assert all(Path(name).suffix not in FORBIDDEN_ASSET_SUFFIXES for name in names)
