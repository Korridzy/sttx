from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import signal
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

PARAKEET_NAMES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
ALL_NAMES = (*PARAKEET_NAMES, "silero_vad.onnx")
REPO_ID = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
SILERO_RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)


def model_module():
    return importlib.import_module("sttx.model")


def write_files(root: Path, names: Sequence[str], prefix: bytes = b"asset-") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(prefix + name.encode())


def bundle_paths(bundle) -> tuple[Path, ...]:
    return (
        bundle.encoder,
        bundle.decoder,
        bundle.joiner,
        bundle.tokens,
        bundle.silero,
    )


def snapshot_fake(
    local_snapshot: Path,
    online_snapshot: Path | None = None,
) -> tuple[Callable[..., str], list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    def download(
        *,
        repo_id: str,
        allow_patterns: list[str],
        local_files_only: bool,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "allow_patterns": tuple(allow_patterns),
                "local_files_only": local_files_only,
            }
        )
        if local_files_only:
            return str(local_snapshot)
        assert online_snapshot is not None
        return str(online_snapshot)

    return download, calls


def silero_writer(payload: bytes, calls: list[Path]) -> Callable[[Path], None]:
    def download(destination: Path) -> None:
        calls.append(destination)
        destination.write_bytes(payload)

    return download


def test_complete_model_dir_is_zero_network(tmp_path: Path) -> None:
    # Given: a complete flat bundle and network seams that fail if reached.
    model = model_module()
    bundle_dir = tmp_path / "bundle"
    write_files(bundle_dir, ALL_NAMES)

    def forbidden_snapshot(**_kwargs) -> str:
        raise AssertionError("Hugging Face network seam was called")

    def forbidden_silero(_destination: Path) -> None:
        raise AssertionError("Silero network seam was called")

    # When: the explicit directory is resolved.
    bundle = model.resolve_bundle(
        bundle_dir,
        _snapshot_download=forbidden_snapshot,
        _silero_downloader=forbidden_silero,
    )

    # Then: exact local paths are returned without consulting either seam.
    assert tuple(path.name for path in bundle_paths(bundle)) == ALL_NAMES
    assert all(path.parent == bundle_dir for path in bundle_paths(bundle))


def test_each_missing_bundle_file_is_environment_error(tmp_path: Path) -> None:
    model = model_module()
    for missing in ALL_NAMES:
        # Given: one required exact file is absent from an otherwise complete bundle.
        bundle_dir = tmp_path / missing.replace(".", "-")
        write_files(bundle_dir, tuple(name for name in ALL_NAMES if name != missing))

        # When/Then: validation identifies the incomplete environment.
        with pytest.raises(model.ModelEnvironmentError) as captured:
            model.resolve_bundle(bundle_dir)
        assert captured.value.path == bundle_dir / missing


@pytest.mark.parametrize("bad_kind", ["empty", "unreadable"])
def test_each_invalid_bundle_file_is_environment_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_kind: str,
) -> None:
    model = model_module()
    for invalid in ALL_NAMES:
        # Given: one exact asset is empty or unreadable.
        bundle_dir = tmp_path / f"{bad_kind}-{invalid.replace('.', '-')}"
        write_files(bundle_dir, ALL_NAMES)
        invalid_path = bundle_dir / invalid
        if bad_kind == "empty":
            invalid_path.write_bytes(b"")
        else:
            real_access = os.access
            monkeypatch.setattr(
                os,
                "access",
                lambda path, mode, *, _bad=invalid_path: (
                    False if Path(path) == _bad else real_access(path, mode)
                ),
            )

        # When/Then: the boundary rejects that exact asset.
        with pytest.raises(model.ModelEnvironmentError) as captured:
            model.resolve_bundle(bundle_dir)
        assert captured.value.path == invalid_path
        monkeypatch.undo()


def test_warm_hf_and_silero_caches_never_call_network(tmp_path: Path) -> None:
    # Given: a complete local HF snapshot and complete Silero final.
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    silero = tmp_path / "cache" / "silero_vad.onnx"
    silero.parent.mkdir()
    silero.write_bytes(b"warm-silero")
    snapshot_download, calls = snapshot_fake(snapshot)
    silero_calls: list[Path] = []

    # When: default resolution is repeated.
    first = model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
        _silero_downloader=silero_writer(b"new", silero_calls),
    )
    second = model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
        _silero_downloader=silero_writer(b"new", silero_calls),
    )

    # Then: each HF lookup is local-only and Silero is never downloaded.
    assert [call["local_files_only"] for call in calls] == [True, True]
    assert silero_calls == []
    assert bundle_paths(first) == bundle_paths(second)
    assert silero.read_bytes() == b"warm-silero"


def test_absent_components_are_acquired_once(tmp_path: Path) -> None:
    # Given: no valid HF snapshot or Silero final.
    model = model_module()
    missing = tmp_path / "missing"
    acquired = tmp_path / "snapshot"
    write_files(acquired, PARAKEET_NAMES)
    silero = tmp_path / "cache" / "silero_vad.onnx"
    snapshot_download, calls = snapshot_fake(missing, acquired)
    silero_calls: list[Path] = []
    download_silero = silero_writer(b"downloaded-silero", silero_calls)

    # When: cold resolution is followed by a warm resolution.
    first = model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
        _silero_downloader=download_silero,
    )
    warm_download, warm_calls = snapshot_fake(acquired)
    second = model.resolve_bundle(
        _snapshot_download=warm_download,
        _silero_cache_path=silero,
        _silero_downloader=download_silero,
    )

    # Then: cold components acquire once and warm components stay local.
    assert [call["local_files_only"] for call in calls] == [True, False]
    assert [call["local_files_only"] for call in warm_calls] == [True]
    assert len(silero_calls) == 1
    assert first.silero.read_bytes() == b"downloaded-silero"
    assert bundle_paths(first) == bundle_paths(second)


def test_zero_length_silero_is_reacquired(tmp_path: Path) -> None:
    # Given: valid Parakeet files and a zero-byte Silero final.
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    silero = tmp_path / "cache" / "silero_vad.onnx"
    silero.parent.mkdir()
    silero.write_bytes(b"")
    snapshot_download, _calls = snapshot_fake(snapshot)
    silero_calls: list[Path] = []

    # When: the bundle is resolved.
    bundle = model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
        _silero_downloader=silero_writer(b"recovered", silero_calls),
    )

    # Then: the invalid final is atomically replaced by non-empty content.
    assert len(silero_calls) == 1
    assert bundle.silero == silero
    assert silero.read_bytes() == b"recovered"


def test_silero_staging_is_unique_cache_sibling(tmp_path: Path) -> None:
    # Given: two cold caches resolved independently.
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    snapshot_download, _calls = snapshot_fake(snapshot)
    seen: list[Path] = []

    # When: each Silero asset is acquired.
    for cache_name in ("cache-a", "cache-b"):
        final = tmp_path / cache_name / "silero_vad.onnx"
        model.resolve_bundle(
            _snapshot_download=snapshot_download,
            _silero_cache_path=final,
            _silero_downloader=silero_writer(b"vad", seen),
        )

    # Then: exclusive staging names differ, share their final parent, and disappear.
    assert len({path.name for path in seen}) == 2
    assert all(path.parent.name in {"cache-a", "cache-b"} for path in seen)
    assert all(not path.exists() for path in seen)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_silero_staging_cleans_on_signal(
    tmp_path: Path,
    interruption: type[BaseException],
) -> None:
    # Given: a pre-existing final and a downloader interrupted after staging bytes.
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    snapshot_download, _calls = snapshot_fake(snapshot)
    final = tmp_path / "cache" / "silero_vad.onnx"
    final.parent.mkdir()
    final.write_bytes(b"")
    seen: list[Path] = []

    def interrupted(destination: Path) -> None:
        seen.append(destination)
        destination.write_bytes(b"partial")
        raise interruption()

    # When/Then: the signal propagates, the final is unchanged, and staging is gone.
    with pytest.raises(interruption):
        model.resolve_bundle(
            _snapshot_download=snapshot_download,
            _silero_cache_path=final,
            _silero_downloader=interrupted,
        )
    assert final.read_bytes() == b""
    assert len(seen) == 1
    assert not seen[0].exists()


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
        allow_patterns: list[str],
        local_files_only: bool,
    ) -> str:
        assert repo_id == REPO_ID
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


def test_incomplete_hf_snapshot_retries_online_then_validates(tmp_path: Path) -> None:
    # Given: a partial local snapshot and an online result that is also incomplete.
    model = model_module()
    partial = tmp_path / "partial"
    write_files(partial, PARAKEET_NAMES[:-1])
    online = tmp_path / "online"
    write_files(online, PARAKEET_NAMES[:-1])
    snapshot_download, calls = snapshot_fake(partial, online)

    # When/Then: online is attempted once but incomplete runtime files are rejected.
    with pytest.raises(model.ModelEnvironmentError) as captured:
        model.resolve_bundle(_snapshot_download=snapshot_download)
    assert [call["local_files_only"] for call in calls] == [True, False]
    assert captured.value.path == online / PARAKEET_NAMES[-1]


def test_silero_download_exception_preserves_final_and_cleans_staging(
    tmp_path: Path,
) -> None:
    # Given: an invalid final and a downloader that fails after writing partial data.
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    snapshot_download, _calls = snapshot_fake(snapshot)
    final = tmp_path / "cache" / "silero_vad.onnx"
    final.parent.mkdir()
    final.write_bytes(b"")
    staged: list[Path] = []

    def broken(destination: Path) -> None:
        staged.append(destination)
        destination.write_bytes(b"partial")
        raise OSError("download failed")

    # When/Then: the typed boundary error leaves no promoted or staged partial.
    with pytest.raises(model.ModelEnvironmentError):
        model.resolve_bundle(
            _snapshot_download=snapshot_download,
            _silero_cache_path=final,
            _silero_downloader=broken,
        )
    assert final.read_bytes() == b""
    assert len(staged) == 1
    assert not staged[0].exists()


def test_no_revision_or_checksum_pin_exists(tmp_path: Path) -> None:
    # Given: an incomplete local cache and valid online floating result.
    model = model_module()
    local = tmp_path / "local"
    online = tmp_path / "online"
    write_files(online, PARAKEET_NAMES)
    silero = tmp_path / "silero_vad.onnx"
    silero.write_bytes(b"vad")
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs) -> str:
        calls.append(kwargs)
        return str(local if kwargs["local_files_only"] else online)

    # When: floating resolution falls back online.
    model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
    )

    # Then: the Hub call has only the contract keys and production has no checksum pin.
    assert calls == [
        {
            "repo_id": REPO_ID,
            "allow_patterns": list(PARAKEET_NAMES),
            "local_files_only": True,
        },
        {
            "repo_id": REPO_ID,
            "allow_patterns": list(PARAKEET_NAMES),
            "local_files_only": False,
        },
    ]
    source = inspect.getsource(model)
    assert "revision=" not in source
    assert "sha256" not in source.lower()
    assert model.SILERO_URL == SILERO_RELEASE_URL
