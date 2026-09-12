from __future__ import annotations

import os
import queue
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .real_pipeline_artifacts import JsonValue, root_manifest
from .real_pipeline_runner import CacheEnv, sanitized_env
from .real_pipeline_signal_process import (
    SIGNAL_TIMEOUT_SECONDS,
    FifoBarrier,
    SignalProbe,
    finish_probe,
    start_cancellable_acquisition,
    start_sttx,
    wait_for_connect,
    wait_for_hf_partial,
)
from sttx.model import (
    PARAKEET_FILENAMES,
    PARAKEET_REPO_ID,
    PARAKEET_REVISION,
    SILERO_URL,
)


@dataclass(frozen=True, slots=True)
class _ConnectProxy:
    url: str
    expected_target: str
    connected: threading.Event
    targets: queue.Queue[str]


@dataclass(frozen=True, slots=True)
class _HeldTransferProxy:
    url: str
    transfer_held: threading.Event


@contextmanager
def _connect_proxy(expected_target: str) -> Generator[_ConnectProxy, None, None]:
    connected = threading.Event()
    release = threading.Event()
    targets: queue.Queue[str] = queue.Queue()

    class ConnectHandler(BaseHTTPRequestHandler):
        def do_CONNECT(self) -> None:  # noqa: N802
            targets.put(self.path)
            connected.set()
            _ = release.wait()

    server = HTTPServer(("127.0.0.1", 0), ConnectHandler)
    server.timeout = SIGNAL_TIMEOUT_SECONDS
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        yield _ConnectProxy(
            url=f"http://127.0.0.1:{server.server_port}",
            expected_target=expected_target,
            connected=connected,
            targets=targets,
        )
    finally:
        release.set()
        server.server_close()
        thread.join()
        assert not thread.is_alive()


@contextmanager
def _held_transfer_proxy(
    threshold: int,
) -> Generator[_HeldTransferProxy, None, None]:
    transfer_held = threading.Event()
    release = threading.Event()
    counter_lock = threading.Lock()
    server_to_client = 0

    class TunnelHandler(BaseHTTPRequestHandler):
        connection: socket.socket

        def do_CONNECT(self) -> None:  # noqa: N802
            nonlocal server_to_client
            host, separator, port_text = self.path.rpartition(":")
            if not separator:
                self.send_error(400, "CONNECT target has no port")
                return
            established = False
            try:
                upstream = socket.create_connection(
                    (host, int(port_text)),
                    timeout=SIGNAL_TIMEOUT_SECONDS,
                )
                with upstream:
                    upstream.settimeout(None)
                    self.send_response(200, "Connection established")
                    self.end_headers()
                    established = True
                    while True:
                        readable, _, _ = select.select(
                            (self.connection, upstream),
                            (),
                            (),
                        )
                        for source in readable:
                            destination = upstream if source is self.connection else self.connection
                            chunk = source.recv(64 * 1024)
                            if not chunk:
                                return
                            destination.sendall(chunk)
                            if source is upstream:
                                with counter_lock:
                                    server_to_client += len(chunk)
                                    hold_here = (
                                        server_to_client >= threshold
                                        and not transfer_held.is_set()
                                    )
                                    if hold_here:
                                        transfer_held.set()
                                if hold_here:
                                    _ = release.wait()
            except (OSError, ValueError):
                if not established:
                    self.send_error(502, "CONNECT upstream failed")

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), TunnelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _HeldTransferProxy(
            url=f"http://127.0.0.1:{server.server_port}",
            transfer_held=transfer_held,
        )
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()
        assert not thread.is_alive()


def prove_signal_barriers(
    tmp_path: Path,
    cold: CacheEnv,
    bundle_dir: Path,
    media: Path,
    evidence: dict[str, JsonValue],
) -> None:
    before = root_manifest(cold.root)
    probes: list[SignalProbe] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        probes.append(_probe_hf(tmp_path, bundle_dir, media, signum))
        probes.append(_probe_silero(tmp_path, bundle_dir, signum))
        probes.append(_probe_cancellable_acquisition(tmp_path, bundle_dir, signum))
        probes.append(_probe_native_decode(tmp_path, cold, bundle_dir, media, signum))
    after = root_manifest(cold.root)
    assert before == after
    for probe in probes:
        expected = 128 + int(signal.Signals[probe.signum])
        assert probe.returncode == expected, probe.to_json()
        assert probe.live_after is False, probe.to_json()
        assert "Traceback" not in probe.stderr, probe.to_json()
    evidence["signals"] = {
        "cold_root_manifest_identical_after_signal_qa": True,
        "probes": [probe.to_json() for probe in probes],
    }


def _probe_hf(
    tmp_path: Path,
    bundle_dir: Path,
    media: Path,
    signum: signal.Signals,
) -> SignalProbe:
    env = sanitized_env(tmp_path / f"hf-{signum.name}")
    env.values["HF_HUB_DISABLE_XET"] = "1"
    silero_final = env.root / "home" / ".cache" / "sttx" / "silero_vad.onnx"
    silero_final.parent.mkdir(parents=True)
    _ = shutil.copy2(bundle_dir / "silero_vad.onnx", silero_final)
    outdir = env.root / "out"
    # huggingface_hub writes response bodies in 10 MiB chunks; hold only after
    # several chunks must have reached a runtime .incomplete file.
    with _held_transfer_proxy(32 * 1024 * 1024) as proxy:
        env.values.update({
            "https_proxy": proxy.url,
            "HTTPS_PROXY": proxy.url,
            "no_proxy": "",
            "NO_PROXY": "",
        })
        process = start_sttx(media, outdir, env)
        try:
            barrier = wait_for_hf_partial(env.root, process, proxy.transfer_held)
            return finish_probe("hf", signum, env.root, process, barrier)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                _ = process.communicate(timeout=5)


def _probe_silero(tmp_path: Path, bundle_dir: Path, signum: signal.Signals) -> SignalProbe:
    root = tmp_path / f"silero-{signum.name}"
    ready = root / "ready.fifo"
    hold = root / "hold.fifo"
    with FifoBarrier.open(ready, "Silero staged download") as ready_barrier:
        os.mkfifo(hold)
        code = (
            "from pathlib import Path\n"
            "import os\n"
            "import select\n"
            "import urllib.request\n"
            "from sttx.audio import PreparedAudio\n"
            "from sttx.cli import RunnerDependencies, run\n"
            "from sttx.model import resolve_bundle, PARAKEET_FILENAMES, PARAKEET_REVISION\n"
            "from sttx.output import Segment, Transcript\n"
            f"root=Path({str(root)!r}); snapshot=root/'snapshot'; snapshot.mkdir(parents=True)\n"
            "media=root/'input.wav'; media.write_bytes(b'wav')\n"
            "for name in PARAKEET_FILENAMES: (snapshot/name).write_bytes((Path("
            f"{str(bundle_dir)!r})/name).read_bytes())\n"
            "def silero(dest):\n"
            f"    with urllib.request.urlopen({SILERO_URL!r}, timeout=60) as response, dest.open('wb') as output:\n"
            "        chunk=response.read(4096)\n"
            "        if not chunk: raise RuntimeError('empty Silero response')\n"
            "        output.write(chunk); output.flush()\n"
            "        os.fsync(output.fileno())\n"
            "        staged_size=dest.stat().st_size\n"
            f"    ready_fd=os.open({str(ready)!r}, os.O_WRONLY)\n"
            "    os.write(ready_fd, f'{dest}\\t{staged_size}\\n'.encode())\n"
            "    os.close(ready_fd)\n"
            f"    hold_fd=os.open({str(hold)!r}, os.O_RDONLY | os.O_NONBLOCK)\n"
            "    select.select((hold_fd,), (), ())\n"
            "def resolver(_model_dir):\n"
            "    def cached_snapshot(*, repo_id, revision, allow_patterns, local_files_only, cache_dir):\n"
            "        assert revision == PARAKEET_REVISION\n"
            "        return str(snapshot)\n"
            "    return resolve_bundle(_snapshot_download=cached_snapshot, _silero_cache_path=root/'cache'/'silero_vad.onnx', _silero_downloader=silero)\n"
            "def transcribe(audio, *, recognizer, vad, progress=None, activity=None):\n"
            "    return Transcript(language='en', duration=1.0, segments=(Segment(id=0, start=0.0, end=1.0, text='ok'),))\n"
            "dependencies=RunnerDependencies(normalize_media=lambda path: PreparedAudio(path=media, sample_count=16000), resolve_bundle=resolver, make_recognizer=lambda bundle: bundle, make_vad=lambda bundle: bundle, transcribe=transcribe)\n"
            "raise SystemExit(run([str(media), '-d', str(root/'out')], _dependencies=dependencies, _cwd=root))\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            staged_path, staged_size = ready_barrier.wait(process).split("\t", maxsplit=1)
            barrier: dict[str, JsonValue] = {
                "ready": staged_path,
                "staged_path": staged_path,
                "staged_size": int(staged_size),
                "official_url": SILERO_URL,
            }
            return finish_probe("silero", signum, root, process, barrier)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                _ = process.communicate(timeout=5)


def _probe_cancellable_acquisition(
    tmp_path: Path,
    bundle_dir: Path,
    signum: signal.Signals,
) -> SignalProbe:
    root = tmp_path / f"cancellable-acquisition-{signum.name}"
    env = sanitized_env(root)
    snapshot = (
        Path(env.values["HF_HUB_CACHE"])
        / f"models--{PARAKEET_REPO_ID.replace('/', '--')}"
        / "snapshots"
        / PARAKEET_REVISION
    )
    snapshot.mkdir(parents=True)
    for name in PARAKEET_FILENAMES:
        _ = shutil.copy2(bundle_dir / name, snapshot / name)

    with _connect_proxy(f"{urlsplit(SILERO_URL).netloc}:443") as proxy:
        process = start_cancellable_acquisition(env, proxy.url)
        try:
            barrier = wait_for_connect(proxy, process, root)
            probe = finish_probe(
                "cancellable_acquisition",
                signum,
                root,
                process,
                barrier,
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                _ = process.communicate(timeout=5)
    remaining = sorted(root.rglob("*.tmp"))
    assert remaining == [], probe.to_json()
    return probe


def _probe_native_decode(
    tmp_path: Path,
    cold: CacheEnv,
    bundle_dir: Path,
    media: Path,
    signum: signal.Signals,
) -> SignalProbe:
    del cold
    env = sanitized_env(tmp_path / f"decode-{signum.name}")
    outdir = env.root / "out"
    entered = env.root / "decode-entered.fifo"
    spent = env.root / "decode_native_us"
    # A Python signal handler cannot run while a native call holds the interpreter,
    # so it fires at the first bytecode boundary after decode_stream returns. The
    # finally clause still runs while SystemExit propagates, which is why the
    # elapsed native time is recorded there: its presence proves the child entered
    # the native call and stayed in it while the signal was already pending.
    with FifoBarrier.open(entered, "native decode entry") as decode_barrier:
        code = (
            "import os\n"
            "import time\n"
            "from pathlib import Path\n"
            "from dataclasses import replace\n"
            "from sttx.cli import _PRODUCTION_DEPENDENCIES, run\n"
            "class AnnouncingRecognizer:\n"
            "    def __init__(self, raw):\n"
            "        self.raw=raw\n"
            "        self.announced=False\n"
            "    def create_stream(self):\n"
            "        return self.raw.create_stream()\n"
            "    def decode_stream(self, stream):\n"
            "        first=not self.announced\n"
            "        if not first:\n"
            "            self.raw.decode_stream(stream)\n"
            "            return\n"
            "        self.announced=True\n"
            f"        entered_fd=os.open({str(entered)!r}, os.O_WRONLY)\n"
            "        os.write(entered_fd, b'decode_entered\\n')\n"
            "        os.close(entered_fd)\n"
            "        started=time.monotonic()\n"
            "        try:\n"
            "            self.raw.decode_stream(stream)\n"
            "        finally:\n"
            f"            Path({str(spent)!r}).write_text("
            "str(int((time.monotonic()-started)*1_000_000)), encoding='utf-8')\n"
            "def make_recognizer(bundle):\n"
            "    return AnnouncingRecognizer(_PRODUCTION_DEPENDENCIES.make_recognizer(bundle))\n"
            "raise SystemExit(run([\n"
            f"    {str(media)!r}, '-d', {str(outdir)!r}, '--model-dir', {str(bundle_dir)!r},\n"
            "], _dependencies=replace(_PRODUCTION_DEPENDENCIES, make_recognizer=make_recognizer)))\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            env=env.subprocess_env(offline=True),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            record = decode_barrier.wait(process)
            assert record == "decode_entered"
            barrier: dict[str, JsonValue] = {"real_decode_entered": record}
            probe = finish_probe("native_decode", signum, env.root, process, barrier)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                _ = process.communicate(timeout=5)
    barrier["native_decode_us"] = (
        int(spent.read_text(encoding="utf-8")) if spent.is_file() else -1
    )
    barrier["output_finals_exist"] = (
        any(outdir.glob("*.json")) or any(outdir.glob("*.txt"))
    )
    assert barrier["native_decode_us"] > 0, probe.to_json()
    assert barrier["output_finals_exist"] is False, probe.to_json()
    return probe
