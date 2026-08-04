from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | Sequence[JsonValue]
    | Mapping[str, JsonValue]
)

RUNTIME_PACKAGES = (
    "huggingface-hub",
    "numpy",
    "pytest",
    "sherpa-onnx",
    "sherpa-onnx-bin",
    "sttx",
)


class BundlePaths(Protocol):
    @property
    def encoder(self) -> Path: ...

    @property
    def decoder(self) -> Path: ...

    @property
    def joiner(self) -> Path: ...

    @property
    def tokens(self) -> Path: ...

    @property
    def silero(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class TaskArtifacts:
    root: Path
    identity: Path
    log: Path
    cleanup: Path
    hash_manifest: Path
    done_claim: Path


def task_artifacts(identity_output: Path | None, fallback_root: Path) -> TaskArtifacts:
    identity = identity_output or fallback_root / "task-10-sttx-python-transcriber.json"
    root = identity.parent
    root.mkdir(parents=True, exist_ok=True)
    return TaskArtifacts(
        root=root,
        identity=identity,
        log=root / "task-10-sttx-python-transcriber.log",
        cleanup=root / "task-10-cleanup-receipt.json",
        hash_manifest=root / "task-10-artifact-hashes.json",
        done_claim=root / "task-10-done-claim.json",
    )


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(UTC).isoformat()} {message}\n")


def write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    with staging.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def json_mapping(value: JsonValue, label: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{label} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_identity(path: Path) -> dict[str, JsonValue]:
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def bundle_identity(bundle: BundlePaths) -> dict[str, JsonValue]:
    return {
        "encoder": asset_identity(bundle.encoder),
        "decoder": asset_identity(bundle.decoder),
        "joiner": asset_identity(bundle.joiner),
        "tokens": asset_identity(bundle.tokens),
        "silero": asset_identity(bundle.silero),
    }


def snapshot_commit(bundle: BundlePaths) -> str:
    snapshots = bundle.encoder.parent.parent
    if snapshots.name != "snapshots":
        raise AssertionError(f"encoder is not inside a HF snapshot: {bundle.encoder}")
    return bundle.encoder.parent.name


def environment_identity() -> dict[str, JsonValue]:
    packages: dict[str, JsonValue] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "utc": datetime.now(UTC).isoformat(),
        "platform": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": sys.version,
        },
        "tools": {
            "poetry": command_version(("poetry", "--version")),
            "ffmpeg": command_version(("ffmpeg", "-version")),
            "sttx": command_version((str(Path(sys.executable).with_name("sttx")), "--version")),
        },
        "packages": packages,
    }


def command_version(command: tuple[str, ...]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return (completed.stdout or completed.stderr).splitlines()[0]


def root_manifest(root: Path) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    if not root.exists():
        return records
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(asset_identity(path))
    return records


def copy_bundle(bundle: BundlePaths, destination: Path) -> dict[str, JsonValue]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = {
        "encoder.int8.onnx": bundle.encoder,
        "decoder.int8.onnx": bundle.decoder,
        "joiner.int8.onnx": bundle.joiner,
        "tokens.txt": bundle.tokens,
        "silero_vad.onnx": bundle.silero,
    }
    for name, source in copied.items():
        shutil.copy2(source, destination / name)
    return {name: asset_identity(destination / name) for name in copied}


def write_hash_manifest(artifacts: TaskArtifacts) -> None:
    files = [
        path
        for path in artifacts.root.glob("task-10*")
        if path.is_file() and path != artifacts.hash_manifest
    ]
    write_json(
        artifacts.hash_manifest,
        {"artifacts": {path.name: asset_identity(path) for path in sorted(files)}},
    )
