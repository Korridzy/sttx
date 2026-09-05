from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import sttx.output as output_module
from sttx.asr import ProgressCallback
from sttx.audio import PreparedAudio, normalize_media
from sttx.asr_events import ActivityCallback
from sttx.cli import RunnerDependencies, run
from sttx.model import (
    PARAKEET_FILENAMES,
    PARAKEET_REVISION,
    ModelBundle,
    ModelEnvironmentError,
    _bundle_from_directory,
    resolve_bundle,
)
from sttx.output import OutputPaths, Segment, Transcript

PHASES: Final = (
    "ffmpeg",
    "hf",
    "silero",
    "decode",
    "before_replace",
    "between_replaces",
)


@dataclass(frozen=True, slots=True)
class CleanupBarrier:
    ready: Path
    release: Path


class BlockingCleanupAudio(PreparedAudio):
    __slots__ = ("_barrier",)

    def __init__(
        self,
        *,
        path: Path,
        sample_count: int,
        barrier: CleanupBarrier,
    ) -> None:
        super().__init__(path=path, sample_count=sample_count)
        self._barrier = barrier

    def cleanup(self) -> None:
        descriptor = os.open(self._barrier.release, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as release:
            self._barrier.ready.write_text("ready", encoding="utf-8")
            select.select((release,), (), ())
            release.read(1)
        super().cleanup()


def _write_assets(root: Path, names: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(f"asset:{name}".encode())


def _bundle(root: Path) -> ModelBundle:
    return ModelBundle(
        encoder=root / PARAKEET_FILENAMES[0],
        decoder=root / PARAKEET_FILENAMES[1],
        joiner=root / PARAKEET_FILENAMES[2],
        tokens=root / PARAKEET_FILENAMES[3],
        silero=root / "silero_vad.onnx",
    )


def _block(ready: Path) -> None:
    ready.write_text("ready", encoding="utf-8")
    while True:
        time.sleep(60)


def _ffmpeg_environment(root: Path, ready: Path) -> None:
    executable = root / "bin" / "ffmpeg"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        "#!/bin/sh\n"
        "trap '' TERM\n"
        f"printf '%s' \"$$\" > '{root / 'ffmpeg.pid'}'\n"
        f"printf '%s' \"${{15}}\" > '{root / 'ffmpeg-temp.path'}'\n"
        f"printf ready > '{ready}'\n"
        "while :; do sleep 60; done\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    os.environ["PATH"] = f"{executable.parent}{os.pathsep}{os.environ['PATH']}"


def _resolve_for_phase(phase: str, root: Path, ready: Path) -> ModelBundle:
    snapshot = root / "snapshot"
    _write_assets(snapshot, PARAKEET_FILENAMES)
    if phase == "hf":
        partial = root / "hf-partial"

        def snapshot_download(
            *,
            repo_id: str,
            revision: str,
            allow_patterns: list[str],
            local_files_only: bool,
            cache_dir: Path | None,
        ) -> str:
            assert cache_dir is None or cache_dir.is_absolute()
            assert revision == PARAKEET_REVISION
            del repo_id, allow_patterns
            if local_files_only:
                return str(root / "missing-snapshot")
            _write_assets(partial, PARAKEET_FILENAMES[:-1])
            _block(ready)
            return str(partial)

        return resolve_bundle(
            _snapshot_download=snapshot_download,
            _silero_cache_path=root / "cache" / "silero_vad.onnx",
        )
    if phase == "silero":

        def download_silero(destination: Path) -> None:
            destination.write_bytes(b"partial-silero")
            _block(ready)

        return resolve_bundle(
            _snapshot_download=lambda **_kwargs: str(snapshot),
            _silero_cache_path=root / "cache" / "silero_vad.onnx",
            _silero_downloader=download_silero,
        )
    return _bundle(snapshot)


def _write_for_phase(
    phase: str,
    transcript: Transcript,
    paths: OutputPaths,
    ready: Path,
) -> None:
    real_replace = output_module.replace_temp
    calls = 0

    def blocking_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if phase == "before_replace" and calls == 1:
            _block(ready)
        real_replace(source, destination)
        if phase == "between_replaces" and calls == 1:
            _block(ready)

    output_module.replace_temp = blocking_replace
    try:
        output_module.write_outputs(transcript, paths)
    finally:
        output_module.replace_temp = real_replace


def _state(root: Path, phase: str, exit_code: int, restored: bool) -> dict[str, str | int | bool | list[str]]:
    partial = root / "hf-partial"
    hf_partial_valid = False
    if partial.exists():
        try:
            _bundle_from_directory(partial, include_silero=False)
        except ModelEnvironmentError:
            pass
        else:
            hf_partial_valid = True
    output_dir = root / "out"
    return {
        "phase": phase,
        "exit_code": exit_code,
        "handlers_restored": restored,
        "json": (output_dir / "episode.json").read_text(encoding="utf-8")
        if (output_dir / "episode.json").exists()
        else "",
        "txt": (output_dir / "episode.txt").read_text(encoding="utf-8")
        if (output_dir / "episode.txt").exists()
        else "",
        "staging": sorted(path.name for path in root.rglob("*.tmp")),
        "prepared_exists": (root / "prepared.wav").exists(),
        "hf_partial_valid": hf_partial_valid,
    }


def _execute(phase: str, root: Path) -> int:
    ready = root / "ready"
    media = root / "input.media"
    media.write_bytes(b"media")
    outdir = root / "out"
    outdir.mkdir()
    (outdir / "episode.json").write_text("old-json", encoding="utf-8")
    (outdir / "episode.txt").write_text("old-txt", encoding="utf-8")
    if phase == "ffmpeg":
        _ffmpeg_environment(root, ready)

    prepared_path = root / "prepared.wav"
    cleanup_release = root / "cleanup-release"
    if phase == "cleanup":
        os.mkfifo(cleanup_release)

    def normalize(_path: Path) -> PreparedAudio:
        prepared_path.write_bytes(b"wav")
        if phase == "cleanup":
            return BlockingCleanupAudio(
                path=prepared_path,
                sample_count=16_000,
                barrier=CleanupBarrier(
                    ready=root / "cleanup-ready",
                    release=cleanup_release,
                ),
            )
        return PreparedAudio(path=prepared_path, sample_count=16_000)

    def transcribe(
        audio: PreparedAudio,
        *,
        recognizer: ModelBundle,
        vad: ModelBundle,
        progress: ProgressCallback | None = None,
        activity: ActivityCallback | None = None,
    ) -> Transcript:
        del audio, recognizer, vad, progress, activity
        if phase in {"cleanup", "decode"}:
            _block(ready)
        return Transcript(
            language="ru",
            duration=1.0,
            segments=(Segment(id=0, start=0.0, end=1.0, text="new text"),),
        )

    def write_outputs(transcript: Transcript, paths: OutputPaths) -> None:
        _write_for_phase(phase, transcript, paths, ready)

    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    code = run(
        [str(media), "-o", "episode", "-d", str(outdir)],
        _dependencies=RunnerDependencies(
            normalize_media=normalize_media if phase == "ffmpeg" else normalize,
            resolve_bundle=lambda _model_dir: _resolve_for_phase(phase, root, ready),
            make_recognizer=lambda bundle: bundle,
            make_vad=lambda bundle: bundle,
            transcribe=transcribe,
        ),
        _write_outputs=write_outputs,
        _cwd=root,
    )
    restored = all(signal.getsignal(signum) == handler for signum, handler in previous.items())
    (root / "result.json").write_text(
        json.dumps(_state(root, phase, code, restored), sort_keys=True),
        encoding="utf-8",
    )
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=(*PHASES, "cleanup"))
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    arguments.root.mkdir(parents=True)
    return _execute(arguments.phase, arguments.root)


if __name__ == "__main__":
    raise SystemExit(main())
