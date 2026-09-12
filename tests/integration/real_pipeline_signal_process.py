# noqa: SIZE_OK because signal process helpers form one fingerprint bound native gate boundary
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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


def wait_for_hf_partial(root: Path, process: subprocess.Popen[str]) -> dict[str, JsonValue]:
    deadline = time.monotonic() + SIGNAL_TIMEOUT_SECONDS
    required = {"encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"}
    while time.monotonic() < deadline:
        hf_roots = (root / "hf" / "hub", root / "hf" / "xet")
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
                if size > 0:
                    partials.append(path)
                    sizes[path] = size
        runtime = [
            path
            for path in partials
            if path.name in required
        ]
        missing = sorted(required - {path.name for path in runtime})
        candidates = {
            path: kind
            for path in partials
            if (kind := _runtime_partial_kind(root, path)) is not None
        }
        if process.poll() is not None:
            raise AssertionError("HF acquisition completed before partial barrier")
        if candidates and missing:
            return {
                "partial_files": [str(path) for path in partials[:10]],
                "partial_asset_candidates": [str(path) for path in candidates],
                "partial_asset_kinds": {
                    str(path): kind for path, kind in candidates.items()
                },
                "partial_asset_sizes": {
                    str(path): sizes[path] for path in candidates
                },
                "process_live_at_barrier": process.poll() is None,
                "runtime_complete_count": len(runtime),
                "missing_runtime_finals": missing,
            }
        time.sleep(0.05)
    process.kill()
    raise AssertionError("HF partial barrier timed out")


def wait_for_path(path: Path, process: subprocess.Popen[str]) -> dict[str, JsonValue]:
    deadline = time.monotonic() + SIGNAL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if path.exists():
            return {"ready": path.read_text(encoding="utf-8")}
        if process.poll() is not None:
            raise AssertionError("signal driver exited before ready barrier")
        time.sleep(0.05)
    process.kill()
    raise AssertionError(f"barrier timed out: {path}")


def wait_for_decode_cpu(
    process: subprocess.Popen[str],
    *,
    marker: Path,
    outdir: Path,
) -> dict[str, JsonValue]:
    deadline = time.monotonic() + SIGNAL_TIMEOUT_SECONDS
    marker_ticks: int | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("decode completed before real-decode barrier")
        if marker.exists() and marker_ticks is None:
            marker_ticks = _cpu_ticks(process.pid)
        current_ticks = _cpu_ticks(process.pid)
        finals_exist = _output_finals_exist(outdir)
        if marker_ticks is not None and current_ticks > marker_ticks and not finals_exist:
            return {
                "real_decode_entered": str(marker),
                "cpu_ticks_before": marker_ticks,
                "cpu_ticks_after": current_ticks,
                "output_finals_exist": finals_exist,
            }
        time.sleep(0.01)
    process.kill()
    raise AssertionError("real-decode CPU barrier timed out")


def _cpu_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    return int(fields[13]) + int(fields[14])


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


def _output_finals_exist(outdir: Path) -> bool:
    return any(outdir.glob("*.json")) or any(outdir.glob("*.txt"))


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
