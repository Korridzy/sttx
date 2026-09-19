from __future__ import annotations

import hashlib
import signal
from pathlib import Path

import pytest

from .model_helpers import (
    PARAKEET_NAMES,
    bundle_paths,
    dying_staging_worker,
    model_module,
    silero_writer,
    snapshot_fake,
    write_files,
)


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
        _silero_expected_sha256=hashlib.sha256(b"warm-silero").hexdigest(),
    )
    second = model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
        _silero_downloader=silero_writer(b"new", silero_calls),
        _silero_expected_sha256=hashlib.sha256(b"warm-silero").hexdigest(),
    )

    # Then: each HF lookup is local-only and Silero is never downloaded.
    assert [call["local_files_only"] for call in calls] == [True, True]
    assert [call["revision"] for call in calls] == [
        "2bda32ec70b097a55adaa07d9a7173915b43cc78",
        "2bda32ec70b097a55adaa07d9a7173915b43cc78",
    ]
    assert silero_calls == []
    assert bundle_paths(first) == bundle_paths(second)
    assert first.source == "cache"
    assert second.source == "cache"
    assert silero.read_bytes() == b"warm-silero"


def test_hf_cache_env_is_passed_to_snapshot_downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    cache = tmp_path / "isolated-hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    snapshot_download, calls = snapshot_fake(snapshot)

    model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=tmp_path / "silero_vad.onnx",
        _silero_downloader=silero_writer(b"silero", []),
        _silero_expected_sha256=hashlib.sha256(b"silero").hexdigest(),
    )

    assert [call["cache_dir"] for call in calls] == [cache]


@pytest.mark.parametrize("warm", [False, True])
def test_hub_adapter_receives_pin_when_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warm: bool,
) -> None:
    from sttx import model

    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    downloader, calls = snapshot_fake(snapshot if warm else tmp_path / "absent", snapshot)
    monkeypatch.setattr(model, "snapshot_download", downloader)

    model.resolve_bundle(
        _silero_cache_path=tmp_path / "silero_vad.onnx",
        _silero_downloader=silero_writer(b"vad", []),
        _silero_expected_sha256=hashlib.sha256(b"vad").hexdigest(),
    )

    assert [call["local_files_only"] for call in calls] == ([True] if warm else [True, False])
    assert all(call["revision"] == "2bda32ec70b097a55adaa07d9a7173915b43cc78" for call in calls)


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
        _silero_expected_sha256=hashlib.sha256(b"downloaded-silero").hexdigest(),
    )
    warm_download, warm_calls = snapshot_fake(acquired)
    second = model.resolve_bundle(
        _snapshot_download=warm_download,
        _silero_cache_path=silero,
        _silero_downloader=download_silero,
        _silero_expected_sha256=hashlib.sha256(b"downloaded-silero").hexdigest(),
    )

    # Then: cold components acquire once and warm components stay local.
    assert [call["local_files_only"] for call in calls] == [True, False]
    assert [call["local_files_only"] for call in warm_calls] == [True]
    assert all(call["revision"] == "2bda32ec70b097a55adaa07d9a7173915b43cc78" for call in calls + warm_calls)
    assert len(silero_calls) == 1
    assert first.silero.read_bytes() == b"downloaded-silero"
    assert bundle_paths(first) == bundle_paths(second)
    assert first.source == "download"
    assert second.source == "cache"


@pytest.mark.parametrize("cached", [b"", b"corrupt-silero"])
@pytest.mark.parametrize("warm_parakeet", [False, True])
def test_invalid_silero_is_reacquired(
    tmp_path: Path, cached: bytes, warm_parakeet: bool,
) -> None:
    # Given: an invalid Silero final and a local or acquired Parakeet snapshot.
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    silero = tmp_path / "cache" / "silero_vad.onnx"
    silero.parent.mkdir()
    silero.write_bytes(cached)
    snapshot_download, _calls = snapshot_fake(
        snapshot if warm_parakeet else tmp_path / "missing", snapshot,
    )
    silero_calls: list[Path] = []

    # When: the bundle is resolved.
    bundle = model.resolve_bundle(
        _snapshot_download=snapshot_download,
        _silero_cache_path=silero,
        _silero_downloader=silero_writer(b"recovered", silero_calls),
        _silero_expected_sha256=hashlib.sha256(b"recovered").hexdigest(),
    )

    # Then: the invalid final is atomically replaced by non-empty content.
    assert len(silero_calls) == 1
    assert bundle.silero == silero
    assert silero.read_bytes() == b"recovered"
    assert bundle.source == ("cache+download" if warm_parakeet else "download")


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
            _silero_expected_sha256=hashlib.sha256(b"vad").hexdigest(),
        )

    # Then: exclusive staging names differ, share their final parent, and disappear.
    assert len({path.name for path in seen}) == 2
    assert all(path.parent.name in {"cache-a", "cache-b"} for path in seen)
    assert all(not path.exists() for path in seen)


def test_cancellable_worker_death_removes_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a cancellable acquisition rooted in an isolated home directory.
    model = model_module()
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".cache" / "sttx"

    # When: the spawned worker dies after creating its owned staging file.
    with pytest.raises(model.ModelEnvironmentError):
        model.resolve_bundle_cancellable(_worker=dying_staging_worker)

    # Then: the parent removes the abandoned staging file.
    assert list(cache_dir.glob("*.tmp")) == []


def test_cancellable_worker_death_preserves_foreign_staging_and_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a foreign staging file and valid final in the isolated cache.
    model = model_module()
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".cache" / "sttx"
    cache_dir.mkdir(parents=True)
    foreign = cache_dir / f".{model.SILERO_FILENAME}.deadbeefdeadbeef.x.tmp"
    final = cache_dir / model.SILERO_FILENAME
    foreign.write_bytes(b"foreign")
    final.write_bytes(b"final")

    # When: another invocation's worker dies with its own staging file.
    with pytest.raises(model.ModelEnvironmentError):
        model.resolve_bundle_cancellable(_worker=dying_staging_worker)

    # Then: the foreign staging file and final asset survive.
    assert foreign.exists()
    assert final.exists()


def test_worker_signal_handler_latches_after_first_signal() -> None:
    # Given: a fresh worker signal handler.
    model = model_module()
    handler = model._make_worker_signal_handler()

    # When: the first signal requests worker shutdown.
    with pytest.raises(SystemExit) as captured:
        handler(signal.SIGTERM, None)

    # Then: its exit code names the signal and a second signal is ignored.
    assert captured.value.code == 128 + signal.SIGTERM
    handler(signal.SIGTERM, None)


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
    final.write_bytes(b"old-corrupt-final")
    seen: list[Path] = []

    def interrupted(destination: Path) -> None:
        seen.append(destination)
        destination.write_bytes(b"partial")
        raise interruption()

    # When/Then: the signal propagates, the final is unchanged, and staging is gone.
    for _attempt in range(2):
        with pytest.raises(interruption):
            model.resolve_bundle(
                _snapshot_download=snapshot_download,
                _silero_cache_path=final,
                _silero_downloader=interrupted,
            )
        assert final.read_bytes() == b"old-corrupt-final"
        assert all(not path.exists() for path in seen)
    assert len(set(seen)) == 2


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


@pytest.mark.parametrize("payload", [b"", b"wrong-download"])
def test_invalid_staging_preserves_final_when_expected_hash_is_injected(
    tmp_path: Path, payload: bytes,
) -> None:
    model = model_module()
    snapshot = tmp_path / "snapshot"
    write_files(snapshot, PARAKEET_NAMES)
    downloader, _calls = snapshot_fake(snapshot)
    final = tmp_path / "silero_vad.onnx"
    final.write_bytes(b"old-final")
    staged: list[Path] = []

    with pytest.raises(model.ModelEnvironmentError):
        model.resolve_bundle(
            _snapshot_download=downloader,
            _silero_cache_path=final,
            _silero_downloader=silero_writer(payload, staged),
            _silero_expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )

    assert final.read_bytes() == b"old-final"
    assert len(staged) == 1
    assert not staged[0].exists()
