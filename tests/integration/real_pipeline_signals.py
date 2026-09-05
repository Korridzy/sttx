from __future__ import annotations

import shutil
import signal
import subprocess
from pathlib import Path

from .real_pipeline_artifacts import JsonValue, root_manifest
from .real_pipeline_runner import CacheEnv, sanitized_env
from .real_pipeline_signal_process import (
    SignalProbe,
    finish_probe,
    start_sttx,
    wait_for_decode_cpu,
    wait_for_hf_partial,
    wait_for_path,
)
from sttx.model import SILERO_URL


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
    shutil.copy2(bundle_dir / "silero_vad.onnx", silero_final)
    outdir = env.root / "out"
    process = start_sttx(media, outdir, env)
    barrier = wait_for_hf_partial(env.root, process)
    return finish_probe("hf", signum, env.root, process, barrier)


def _probe_silero(tmp_path: Path, bundle_dir: Path, signum: signal.Signals) -> SignalProbe:
    root = tmp_path / f"silero-{signum.name}"
    ready = root / "ready"
    code = (
        "from pathlib import Path\n"
        "import time\n"
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
        "        import os\n"
        "        os.fsync(output.fileno())\n"
        "        staged_size=dest.stat().st_size\n"
        f"    Path({str(ready)!r}).write_text(f'{{dest}}\\n{{staged_size}}', encoding='utf-8')\n"
        "    while True: time.sleep(60)\n"
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
        [str(Path.cwd() / ".venv" / "bin" / "python"), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    barrier = wait_for_path(ready, process)
    ready_value = barrier["ready"]
    assert isinstance(ready_value, str)
    staged_path, staged_size = ready_value.splitlines()
    barrier["ready"] = staged_path
    barrier["staged_path"] = staged_path
    barrier["staged_size"] = int(staged_size)
    barrier["official_url"] = SILERO_URL
    return finish_probe("silero", signum, root, process, barrier)


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
    ready = env.root / "import-ready"
    marker = env.root / "real_decode_entered"
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "from sttx.asr import transcribe\n"
        "from dataclasses import replace\n"
        "from sttx.cli import _PRODUCTION_DEPENDENCIES, run\n"
        f"Path({str(ready)!r}).write_text('ready', encoding='utf-8')\n"
        "def wrapped_transcribe(audio, *, recognizer, vad, progress=None, activity=None):\n"
        f"    marker = Path({str(marker)!r})\n"
        "    marker.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with marker.open('w', encoding='utf-8') as stream:\n"
        "        stream.write('real_decode_entered')\n"
        "        stream.flush()\n"
        "        os.fsync(stream.fileno())\n"
        "    return transcribe(audio, recognizer=recognizer, vad=vad, progress=progress, activity=activity)\n"
        "raise SystemExit(run([\n"
        f"    {str(media)!r}, '-d', {str(outdir)!r}, '--model-dir', {str(bundle_dir)!r},\n"
        "], _dependencies=replace(_PRODUCTION_DEPENDENCIES, transcribe=wrapped_transcribe)))\n"
    )
    process = subprocess.Popen(
        [str(Path.cwd() / ".venv" / "bin" / "python"), "-c", code],
        env=env.subprocess_env(offline=True),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    import_barrier = wait_for_path(ready, process)
    barrier = {
        **import_barrier,
        **wait_for_decode_cpu(
            process,
            marker=marker,
            outdir=outdir,
        ),
    }
    return finish_probe("native_decode", signum, env.root, process, barrier)
