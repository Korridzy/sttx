from __future__ import annotations

import os
from pathlib import Path

import pytest

from .model_helpers import ALL_NAMES, bundle_paths, model_module, write_files


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
