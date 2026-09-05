from __future__ import annotations

import hashlib
import inspect
import json
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .model_helpers import (
    PARAKEET_NAMES,
    REPO_ID,
    SILERO_RELEASE_URL,
    model_module,
    write_files,
)


def _inventory(root: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
        elif path.is_dir():
            records.append({"path": relative, "type": "dir", "size": 0})
        else:
            payload = path.read_bytes()
            records.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return records


def _sanitized_hf_environment(isolated: Path, external: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "HUGGINGFACE_HUB_CACHE",
        "HUGGINGFACE_ASSETS_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "HOME": str(isolated / "home"),
            "XDG_CACHE_HOME": str(isolated / "xdg"),
            "HF_HOME": str(isolated / "hf"),
            "HF_HUB_CACHE": str(isolated / "hf" / "hub"),
            "HF_XET_CACHE": str(isolated / "hf" / "xet"),
            "HF_ASSETS_CACHE": str(isolated / "hf" / "assets"),
            "HF_TOKEN_PATH": str(isolated / "hf" / "token"),
            "STTX_EXTERNAL_SENTINEL": str(external),
        }
    )
    return environment


def test_hf_signal_states_preserve_finals_and_reject_partial_cache(
    tmp_path: Path,
) -> None:
    # Given: a complete cached final plus an isolated child acquisition root.
    model = model_module()
    isolated = tmp_path / "isolated"
    external = tmp_path / "external"
    write_files(external, ("sentinel",), b"outside-")
    complete = isolated / "hf" / "hub" / "models--owner--repo" / "snapshots" / "complete"
    write_files(complete, PARAKEET_NAMES, b"complete-")
    before = {path.name: path.read_bytes() for path in complete.iterdir()}
    child = """
import os
import signal
from pathlib import Path

hub = Path(os.environ["HF_HUB_CACHE"])
repo = hub / "models--owner--repo"
(hub.parent / "xet" / "logs").mkdir(parents=True, exist_ok=True)
(hub.parent / "xet" / "logs" / "session.log").write_bytes(b"xet")
(repo / "blobs").mkdir(parents=True, exist_ok=True)
(repo / "blobs" / "new.incomplete").write_bytes(b"partial")
(repo / "refs").mkdir(parents=True, exist_ok=True)
(repo / "refs" / "main").write_text("partial")
(repo / "snapshots" / "partial").mkdir(parents=True, exist_ok=True)
(hub / ".locks" / "models--owner--repo").mkdir(parents=True, exist_ok=True)
(hub / ".locks" / "models--owner--repo" / "new.lock").write_bytes(b"")
os.kill(os.getpid(), signal.SIGTERM)
"""

    # When: the sanitized child is interrupted after creating HF-managed resume state.
    completed = subprocess.run(
        [sys.executable, "-c", child],
        env=_sanitized_hf_environment(isolated, external),
        timeout=5,
        check=False,
    )

    # Then: complete finals and external sentinels survive; partial state is rejected.
    assert completed.returncode == -signal.SIGTERM
    assert {path.name: path.read_bytes() for path in complete.iterdir()} == before
    assert (external / "sentinel").read_bytes() == b"outside-sentinel"
    inventory = _inventory(isolated)
    assert inventory
    assert all(
        (
            "/.locks/" in f"/{record['path']}/"
            or "/blobs/" in f"/{record['path']}/"
            or "/snapshots/" in f"/{record['path']}/"
            or "/refs/" in f"/{record['path']}/"
            or "/xet/" in f"/{record['path']}/"
            or record["type"] == "dir"
        )
        for record in inventory
    )
    partial = isolated / "hf" / "hub" / "models--owner--repo" / "snapshots" / "partial"
    online_calls: list[bool] = []

    def interrupted_snapshot(
        *,
        repo_id: str,
        revision: str,
        allow_patterns: list[str],
        local_files_only: bool,
        cache_dir: Path | None = None,
    ) -> str:
        assert cache_dir is None
        assert repo_id == REPO_ID
        assert revision == "2bda32ec70b097a55adaa07d9a7173915b43cc78"
        assert tuple(allow_patterns) == PARAKEET_NAMES
        online_calls.append(local_files_only)
        if local_files_only:
            return str(partial)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        model.resolve_bundle(_snapshot_download=interrupted_snapshot)
    assert online_calls == [True, False]
    shutil.rmtree(isolated)
    shutil.rmtree(external)
    assert not isolated.exists()
    assert not external.exists()


def test_hf_child_environment_cannot_escape_isolated_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: every parent HF variable points at a distinct external sentinel.
    isolated = tmp_path / "isolated"
    external = tmp_path / "external"
    variable_names = (
        "HUGGINGFACE_HUB_CACHE",
        "HUGGINGFACE_ASSETS_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_ASSETS_CACHE",
        "HF_TOKEN_PATH",
    )
    sentinels: dict[str, Path] = {}
    for name in variable_names:
        sentinel = external / name
        sentinel.mkdir(parents=True)
        (sentinel / "sentinel").write_bytes(name.encode())
        monkeypatch.setenv(name, str(sentinel))
        sentinels[name] = sentinel
    child_environment = _sanitized_hf_environment(isolated, external)
    child_environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    child = """
import json
import os
from pathlib import Path
import huggingface_hub.constants as constants

root = Path(os.environ["HF_HOME"]).parent
paths = {
    "HF_HOME": constants.HF_HOME,
    "HF_HUB_CACHE": constants.HF_HUB_CACHE,
    "HF_XET_CACHE": constants.HF_XET_CACHE,
    "HF_ASSETS_CACHE": constants.HF_ASSETS_CACHE,
    "HF_TOKEN_PATH": constants.HF_TOKEN_PATH,
}
for value in paths.values():
    path = Path(value)
    assert path == root or root in path.parents
(Path(paths["HF_HUB_CACHE"]) / "probe").mkdir(parents=True, exist_ok=True)
print(json.dumps(paths, sort_keys=True))
"""

    # When: Hugging Face is imported in the sanitized child.
    completed = subprocess.run(
        [sys.executable, "-c", child],
        env=child_environment,
        timeout=5,
        check=True,
        capture_output=True,
        text=True,
    )

    # Then: resolved paths remain isolated and every external sentinel is unchanged.
    resolved = json.loads(completed.stdout)
    assert all(isolated in Path(value).parents for value in resolved.values())
    assert {
        name: (path / "sentinel").read_bytes()
        for name, path in sentinels.items()
    } == {name: name.encode() for name in variable_names}
    shutil.rmtree(isolated)
    shutil.rmtree(external)
    assert not isolated.exists()
    assert not external.exists()


def test_exact_revision_when_local_resolution_falls_back_online(tmp_path: Path) -> None:
    # Given: an incomplete local cache and valid online result.
    model = model_module()
    local = tmp_path / "local"
    online = tmp_path / "online"
    write_files(online, PARAKEET_NAMES)
    silero = tmp_path / "silero_vad.onnx"
    silero.write_bytes(b"vad")
    calls: list[dict[str, str | list[str] | bool | Path | None]] = []

    def snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        return str(local if kwargs["local_files_only"] else online)

    # When: resolution falls back online.
    model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
    )

    # Then: the Hub call has only the contract keys and production has no checksum pin.
    assert calls == [
        {
            "repo_id": REPO_ID,
            "revision": "2bda32ec70b097a55adaa07d9a7173915b43cc78",
            "allow_patterns": list(PARAKEET_NAMES),
            "local_files_only": True,
            "cache_dir": None,
        },
        {
            "repo_id": REPO_ID,
            "revision": "2bda32ec70b097a55adaa07d9a7173915b43cc78",
            "allow_patterns": list(PARAKEET_NAMES),
            "local_files_only": False,
            "cache_dir": None,
        },
    ]


def test_no_checksum_pin_exists() -> None:
    model = model_module()
    source = inspect.getsource(model)
    assert "sha256" not in source.lower()
    assert model.SILERO_URL == SILERO_RELEASE_URL
