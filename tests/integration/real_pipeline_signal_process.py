# noqa: SIZE_OK because signal process helpers form one fingerprint bound native gate boundary
from __future__ import annotations

import os
import queue
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol

from .real_pipeline_artifacts import JsonValue
from .real_pipeline_runner import CacheEnv, STTX_BIN

SIGNAL_TIMEOUT_SECONDS = 30.0


class ConnectProxy(Protocol):
    @property
    def expected_target(self) -> str: ...

    @property
    def connected(self) -> threading.Event: ...

    @property
    def targets(self) -> queue.Queue[str]: ...


@dataclass(slots=True)
class FifoBarrier:
    path: Path
    name: str
    descriptor: int

    @classmethod
    def open(cls, path: Path, name: str) -> FifoBarrier:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        return cls(path=path, name=name, descriptor=descriptor)

    def __enter__(self) -> FifoBarrier:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        os.close(self.descriptor)

    def wait(
        self,
        process: subprocess.Popen[str],
        *,
        timeout: float = SIGNAL_TIMEOUT_SECONDS,
    ) -> str:
        if process.poll() is not None:
            raise AssertionError(f"{self.name}: child exited first ({process.returncode})")
        deadline = time.monotonic() + timeout
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([self.descriptor], [], [], remaining)
        if not readable:
            if process.poll() is not None:
                raise AssertionError(f"{self.name}: child exited first ({process.returncode})")
            process.kill()
            raise AssertionError(f"{self.name}: barrier timed out")

        record = bytearray()
        while b"\n" not in record:
            try:
                chunk = os.read(self.descriptor, 4096)
            except BlockingIOError as error:
                process.kill()
                raise AssertionError(f"{self.name}: incomplete barrier record") from error
            if not chunk:
                if process.poll() is not None:
                    raise AssertionError(
                        f"{self.name}: child exited first ({process.returncode})"
                    )
                process.kill()
                raise AssertionError(f"{self.name}: FIFO closed before a complete record")
            record.extend(chunk)
        line, _, _ = record.partition(b"\n")
        try:
            return line.decode("utf-8")
        except UnicodeDecodeError as error:
            process.kill()
            raise AssertionError(f"{self.name}: barrier record is not UTF-8") from error


@dataclass(frozen=True, slots=True)
class SignalProbe:
    phase: str
    signum: str
    root: Path
    returncode: int
    barrier: dict[str, JsonValue]
    stdout: str
    stderr: str
    staging: list[str]
    live_after: bool

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "phase": self.phase,
            "signum": self.signum,
            "root": str(self.root),
            "returncode": self.returncode,
            "barrier": self.barrier,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "staging": self.staging,
            "live_after": self.live_after,
        }


def start_sttx(
    media: Path,
    outdir: Path,
    env: CacheEnv,
    *,
    model_dir: Path | None = None,
) -> subprocess.Popen[str]:
    args = [str(STTX_BIN), str(media), "-d", str(outdir)]
    if model_dir is not None:
        args.extend(["--model-dir", str(model_dir)])
    return subprocess.Popen(
        args,
        env=env.subprocess_env(offline=model_dir is not None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def start_cancellable_acquisition(env: CacheEnv, proxy_url: str) -> subprocess.Popen[str]:
    code = (
        "import signal\n"
        "from sttx.model import resolve_bundle_cancellable\n"
        "def exit_on_signal(signum, _frame):\n"
        "    raise SystemExit(128 + signum)\n"
        "signal.signal(signal.SIGINT, exit_on_signal)\n"
        "signal.signal(signal.SIGTERM, exit_on_signal)\n"
        "resolve_bundle_cancellable()\n"
    )
    process_env = env.subprocess_env(offline=True)
    process_env.update({
        "https_proxy": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "no_proxy": "",
        "NO_PROXY": "",
    })
    return subprocess.Popen(
        [sys.executable, "-c", code],
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def finish_probe(
    phase: str,
    signum: signal.Signals,
    root: Path,
    process: subprocess.Popen[str],
    barrier: dict[str, JsonValue],
) -> SignalProbe:
    os.killpg(process.pid, signum)
    try:
        stdout, stderr = process.communicate(timeout=SIGNAL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=5)
        return SignalProbe(
            phase=phase,
            signum=signum.name,
            root=root,
            returncode=-999,
            barrier={
                **barrier,
                "timeout": True,
                "timeout_stdout": _timeout_text(error.stdout),
                "timeout_stderr": _timeout_text(error.stderr),
            },
            stdout=stdout,
            stderr=stderr,
            staging=sorted(path.name for path in root.rglob("*.tmp")),
            live_after=_is_live(process.pid),
        )
    return SignalProbe(
        phase=phase,
        signum=signum.name,
        root=root,
        returncode=process.returncode,
        barrier=barrier,
        stdout=stdout,
        stderr=stderr,
        staging=sorted(path.name for path in root.rglob("*.tmp")),
        live_after=_is_live(process.pid),
    )


def wait_for_connect(
    proxy: ConnectProxy,
    process: subprocess.Popen[str],
    root: Path,
) -> dict[str, JsonValue]:
    if not proxy.connected.wait(timeout=SIGNAL_TIMEOUT_SECONDS):
        if process.poll() is not None:
            raise AssertionError("acquisition driver exited before CONNECT barrier")
        raise AssertionError("acquisition CONNECT barrier timed out")
    target = proxy.targets.get_nowait()
    if target != proxy.expected_target:
        raise AssertionError(f"unexpected CONNECT target: {target}")
    staging = sorted(root.rglob("*.tmp"))
    return {
        "connect_target": target,
        "process_live_at_barrier": process.poll() is None,
        "staging_at_barrier": [str(path) for path in staging],
    }


def wait_for_hf_partial(
    root: Path,
    process: subprocess.Popen[str],
    transfer_held: threading.Event,
) -> dict[str, JsonValue]:
    if not transfer_held.wait(timeout=SIGNAL_TIMEOUT_SECONDS):
        if process.poll() is not None:
            raise AssertionError("HF acquisition child exited before held-transfer barrier")
        process.kill()
        raise AssertionError("HF held-transfer barrier timed out")

    required = {"encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"}
    hf_roots = (root / "hf" / "hub", root / "hf" / "xet")
    observed_files: list[tuple[Path, int]] = []
    partials: list[Path] = []
    sizes: dict[Path, int] = {}
    for hf_root in hf_roots:
        if not hf_root.exists():
            continue
        for path in hf_root.rglob("*"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            observed_files.append((path, size))
            if size > 0:
                partials.append(path)
                sizes[path] = size
    runtime = [path for path in partials if path.name in required]
    missing = sorted(required - {path.name for path in runtime})
    candidates = {
        path: kind
        for path in partials
        if (kind := _runtime_partial_kind(root, path)) is not None
    }
    if process.poll() is not None:
        raise AssertionError("HF acquisition child exited at held-transfer barrier")
    if not candidates:
        listing = ", ".join(
            f"{path} ({size} bytes)" for path, size in observed_files
        )
        raise AssertionError(
            f"HF held transfer has no runtime partial candidate; HF files: [{listing}]"
        )
    if not missing:
        raise AssertionError("HF held transfer has no missing runtime final")
    return {
        "partial_files": [str(path) for path in partials[:10]],
        "partial_asset_candidates": [str(path) for path in candidates],
        "partial_asset_kinds": {str(path): kind for path, kind in candidates.items()},
        "partial_asset_sizes": {str(path): sizes[path] for path in candidates},
        "process_live_at_barrier": process.poll() is None,
        "runtime_complete_count": len(runtime),
        "missing_runtime_finals": missing,
    }


def _runtime_partial_kind(root: Path, path: Path) -> str | None:
    parts = path.resolve().relative_to(root.resolve()).parts
    if path.name.endswith(".incomplete"):
        return "hub_incomplete"
    if "xet" not in parts or path.name == "CACHEDIR.TAG":
        return None
    if any(part in parts for part in ("logs", "metadata", "refs", "snapshots")):
        return None
    if any(part in parts for part in ("chunk-cache", "chunks", "data", "staging")):
        return "xet_data_partial"
    return None


def _is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _timeout_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
