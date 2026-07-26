from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


GIT_TIMEOUT_SECONDS = 10


class ManifestEntry(TypedDict, total=False):
    mode: int
    path: str
    sha256: str
    size: int
    target: str
    type: str


def git_bytes(repository: Path, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
    return completed.stdout


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(repository: Path, relative_path: str) -> ManifestEntry:
    absolute_path = repository / relative_path
    metadata = absolute_path.lstat()
    entry: ManifestEntry = {
        "mode": stat.S_IMODE(metadata.st_mode),
        "path": relative_path.replace(os.sep, "/"),
        "size": metadata.st_size,
    }
    if stat.S_ISREG(metadata.st_mode):
        entry["sha256"] = file_digest(absolute_path)
        entry["type"] = "file"
    elif stat.S_ISDIR(metadata.st_mode):
        entry["type"] = "directory"
    elif stat.S_ISLNK(metadata.st_mode):
        entry["target"] = os.readlink(absolute_path)
        entry["type"] = "symlink"
    elif stat.S_ISFIFO(metadata.st_mode):
        entry["type"] = "fifo"
    elif stat.S_ISSOCK(metadata.st_mode):
        entry["type"] = "socket"
    elif stat.S_ISCHR(metadata.st_mode):
        entry["type"] = "character-device"
    elif stat.S_ISBLK(metadata.st_mode):
        entry["type"] = "block-device"
    else:
        entry["type"] = "unknown"
    return entry


def build_manifest(repository: Path) -> list[ManifestEntry]:
    relative_paths: list[str] = []
    for current_root, directory_names, file_names in os.walk(
        repository,
        topdown=True,
        followlinks=False,
    ):
        relative_root = os.path.relpath(current_root, repository)
        if relative_root == ".":
            directory_names[:] = [
                name for name in directory_names if name not in (".git", ".omo")
            ]
        for name in directory_names + file_names:
            relative_path = (
                name
                if relative_root == "."
                else os.path.join(relative_root, name)
            )
            relative_paths.append(relative_path)
    return [
        manifest_entry(repository, relative_path)
        for relative_path in sorted(relative_paths)
    ]


def canonical_snapshot(repository: Path) -> bytes:
    payload = {
        "manifest": build_manifest(repository),
        "staged_diff_b64": base64.b64encode(
            git_bytes(repository, ["diff", "--cached", "--binary"]),
        ).decode("ascii"),
        "status_porcelain_v2_z_b64": base64.b64encode(
            git_bytes(
                repository,
                ["status", "--porcelain=v2", "-z", "--untracked-files=all"],
            ),
        ).decode("ascii"),
        "unstaged_diff_b64": base64.b64encode(
            git_bytes(repository, ["diff", "--binary"]),
        ).decode("ascii"),
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def write_exclusive_read_only(output_path: Path, payload: bytes) -> None:
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: repo_snapshot.py REPOSITORY OUTPUT")
    repository = Path(sys.argv[1]).resolve()
    payload = canonical_snapshot(repository)
    write_exclusive_read_only(Path(sys.argv[2]), payload)
    print(hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
