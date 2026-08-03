from __future__ import annotations

import os
import multiprocessing
import queue
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, assert_never

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
)

PARAKEET_REPO_ID: Final = (
    "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
)
PARAKEET_FILENAMES: Final = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
SILERO_FILENAME: Final = "silero_vad.onnx"
SILERO_URL: Final = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)
DOWNLOAD_TIMEOUT_SECONDS: Final = 60.0


class SnapshotDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        allow_patterns: list[str],
        local_files_only: bool,
        cache_dir: Path | None,
    ) -> str: ...


SileroDownloader = Callable[[Path], None]
ModelSource: TypeAlias = Literal[
    "explicit",
    "cache",
    "download",
    "cache+download",
    "unknown",
]
BundleWire: TypeAlias = tuple[str, str, str, str, str, ModelSource]
BundleMessage: TypeAlias = (
    tuple[Literal["ok"], BundleWire] | tuple[Literal["error"], str]
)


@dataclass(frozen=True, slots=True)
class ModelEnvironmentError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"cannot resolve model asset {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ModelBundle:
    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path
    silero: Path
    source: ModelSource = "unknown"


def resolve_bundle(
    model_dir: Path | None = None,
    *,
    _snapshot_download: SnapshotDownloader | None = None,
    _silero_cache_path: Path | None = None,
    _silero_downloader: SileroDownloader | None = None,
) -> ModelBundle:
    if model_dir is not None:
        return replace(
            _bundle_from_directory(model_dir, include_silero=True),
            source="explicit",
        )

    download_snapshot = _snapshot_download or _download_snapshot
    cache_dir = _hf_cache_dir()
    try:
        local_snapshot = Path(
            download_snapshot(
                repo_id=PARAKEET_REPO_ID,
                allow_patterns=list(PARAKEET_FILENAMES),
                local_files_only=True,
                cache_dir=cache_dir,
            )
        )
        parakeet = _bundle_from_directory(local_snapshot, include_silero=False)
        parakeet_source: ModelSource = "cache"
    except (
        IncompleteSnapshotError,
        LocalEntryNotFoundError,
        ModelEnvironmentError,
    ):
        try:
            online_snapshot = Path(
                download_snapshot(
                    repo_id=PARAKEET_REPO_ID,
                    allow_patterns=list(PARAKEET_FILENAMES),
                    local_files_only=False,
                    cache_dir=cache_dir,
                )
            )
        except (HfHubHTTPError, OSError) as error:
            raise ModelEnvironmentError(
                path=Path(PARAKEET_REPO_ID),
                reason=f"Hugging Face acquisition failed: {error}",
            ) from error
        parakeet = _bundle_from_directory(online_snapshot, include_silero=False)
        parakeet_source = "download"

    silero_path = _silero_cache_path or (
        Path.home() / ".cache" / "sttx" / SILERO_FILENAME
    )
    silero_source: ModelSource = (
        "cache" if _is_readable_asset(silero_path) else "download"
    )
    silero = _resolve_silero(
        silero_path,
        _silero_downloader or _download_silero,
    )
    source: ModelSource = (
        parakeet_source
        if parakeet_source == silero_source
        else "cache+download"
    )
    return replace(parakeet, silero=silero, source=source)


def resolve_bundle_cancellable(model_dir: Path | None = None) -> ModelBundle:
    if model_dir is not None:
        return resolve_bundle(model_dir)
    context = multiprocessing.get_context("spawn")
    messages: Queue[BundleMessage] = context.Queue()
    process = context.Process(target=_resolve_bundle_worker, args=(messages,))
    process.start()
    try:
        while process.is_alive():
            try:
                return _bundle_from_message(messages.get(timeout=0.1))
            except queue.Empty:
                pass
        try:
            return _bundle_from_message(messages.get_nowait())
        except queue.Empty as error:
            raise ModelEnvironmentError(
                path=Path(PARAKEET_REPO_ID),
                reason=f"model acquisition process exited {process.exitcode}",
            ) from error
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        messages.close()


def _resolve_bundle_worker(messages: Queue[BundleMessage]) -> None:
    try:
        bundle = resolve_bundle()
    except ModelEnvironmentError as error:
        messages.put(("error", str(error)))
        return
    paths: BundleWire = (
        str(bundle.encoder),
        str(bundle.decoder),
        str(bundle.joiner),
        str(bundle.tokens),
        str(bundle.silero),
        bundle.source,
    )
    messages.put(("ok", paths))


def _bundle_from_message(message: BundleMessage) -> ModelBundle:
    match message:
        case ("ok", (encoder, decoder, joiner, tokens, silero, source)):
            return ModelBundle(
                Path(encoder),
                Path(decoder),
                Path(joiner),
                Path(tokens),
                Path(silero),
                source,
            )
        case ("error", reason):
            raise ModelEnvironmentError(path=Path(PARAKEET_REPO_ID), reason=reason)
        case _ as unreachable:
            assert_never(unreachable)


def _download_snapshot(
    *,
    repo_id: str,
    allow_patterns: list[str],
    local_files_only: bool,
    cache_dir: Path | None,
) -> str:
    result = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    if isinstance(result, str):
        return result
    raise ModelEnvironmentError(
        path=Path(repo_id),
        reason="Hugging Face returned download metadata instead of a snapshot",
    )


def _hf_cache_dir() -> Path | None:
    return Path(raw_cache) if (raw_cache := os.environ.get("HF_HUB_CACHE")) else None


def _bundle_from_directory(directory: Path, *, include_silero: bool) -> ModelBundle:
    filenames = [
        *PARAKEET_FILENAMES,
        *([SILERO_FILENAME] if include_silero else []),
    ]
    paths = {name: _require_asset(directory / name) for name in filenames}
    silero = paths.get(SILERO_FILENAME, directory / SILERO_FILENAME)
    return ModelBundle(
        encoder=paths[PARAKEET_FILENAMES[0]],
        decoder=paths[PARAKEET_FILENAMES[1]],
        joiner=paths[PARAKEET_FILENAMES[2]],
        tokens=paths[PARAKEET_FILENAMES[3]],
        silero=silero,
    )


def _require_asset(path: Path) -> Path:
    if not _is_readable_asset(path):
        raise ModelEnvironmentError(
            path=path,
            reason="required file is missing, unreadable, or empty",
        )
    return path


def _is_readable_asset(path: Path) -> bool:
    try:
        return (
            path.is_file()
            and os.access(path, os.R_OK)
            and path.stat().st_size > 0
        )
    except OSError:
        return False


def _resolve_silero(
    final_path: Path,
    downloader: SileroDownloader,
) -> Path:
    if _is_readable_asset(final_path):
        return _require_asset(final_path)

    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_staging = tempfile.mkstemp(
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            dir=final_path.parent,
        )
        os.close(descriptor)
    except OSError as error:
        raise ModelEnvironmentError(
            path=final_path,
            reason=f"cannot allocate Silero staging file: {error}",
        ) from error

    staging = Path(raw_staging)
    try:
        downloader(staging)
        with staging.open("rb+") as staged_file:
            staged_file.flush()
            os.fsync(staged_file.fileno())
        _require_asset(staging)
        os.replace(staging, final_path)
        return _require_asset(final_path)
    except OSError as error:
        raise ModelEnvironmentError(
            path=final_path,
            reason=f"Silero acquisition failed: {error}",
        ) from error
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError as error:
            raise ModelEnvironmentError(
                path=staging,
                reason=f"cannot clean Silero staging file: {error}",
            ) from error


def _download_silero(destination: Path) -> None:
    with urllib.request.urlopen(  # noqa: S310
        SILERO_URL,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)
