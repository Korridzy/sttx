from __future__ import annotations

from pathlib import Path

import pytest

from .model_helpers import (
    PARAKEET_NAMES,
    bundle_paths,
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
    )

    assert [call["cache_dir"] for call in calls] == [cache]


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
    assert first.source == "download"
    assert second.source == "cache"


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
