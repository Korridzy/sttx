from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path

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
SnapshotCall = dict[str, str | tuple[str, ...] | bool | Path | None]


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
) -> tuple[Callable[..., str], list[SnapshotCall]]:
    calls: list[SnapshotCall] = []

    def download(
        *,
        repo_id: str,
        allow_patterns: list[str],
        local_files_only: bool,
        cache_dir: Path | None = None,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "allow_patterns": tuple(allow_patterns),
                "local_files_only": local_files_only,
                "cache_dir": cache_dir,
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
