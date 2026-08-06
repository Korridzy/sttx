from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from .real_pipeline_artifacts import (
    JsonValue,
    bundle_identity,
    json_mapping,
    snapshot_commit,
)
from .real_pipeline_media import (
    read_transcript,
    write_no_audio_mp4,
    write_silence_wav,
    write_two_utterance_wav,
)
from .real_pipeline_observability import (
    observe_hard_split,
    observe_vad_segments,
    write_vad_positive_continuous_wav,
)
from .real_pipeline_runner import (
    STTX_BIN,
    CacheEnv,
    poisoned_env,
    run_command,
    run_sttx,
    trace_command,
)
from .real_pipeline_trace import assert_clean_trace
from sttx.model import ModelBundle

REQUIRED_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
    "silero_vad.onnx",
)


def prove_warm_offline(
    tmp_path: Path,
    artifact_root: Path,
    cold: CacheEnv,
    en_wav: Path,
    evidence: dict[str, JsonValue],
) -> None:
    prefix = artifact_root / "task-10-warm-network.trace"
    outdir = tmp_path / "warm-offline"
    command = trace_command(prefix, (str(STTX_BIN), str(en_wav), "-d", str(outdir)), network_only=True)
    result = run_command(command, env=cold.subprocess_env(offline=True), cwd=Path.cwd(), timeout=1200.0)
    assert result.returncode == 0, result.to_json()
    assert_clean_trace(prefix)
    check = read_transcript(*stdout_paths(result.stdout))
    assert check.text.strip()
    from sttx.model import PARAKEET_REPO_ID, resolve_bundle

    warm_bundle = resolve_bundle()
    warm_identity = {
        "repo_id": PARAKEET_REPO_ID,
        "commit": snapshot_commit(warm_bundle),
        "assets": bundle_identity(warm_bundle),
    }
    cold_model = evidence["model"]
    assert isinstance(cold_model, dict)
    cold_assets = cold_model["assets"]
    assert isinstance(cold_assets, dict)
    assert warm_identity["commit"] == cold_model["commit"]
    assert warm_identity["assets"] == cold_assets
    evidence["commands"] = {
        **json_mapping(evidence["commands"], "commands"),
        "warm_offline": result.to_json(),
    }
    evidence["checks"] = {
        **json_mapping(evidence["checks"], "checks"),
        "warm_offline": asdict(check),
    }
    evidence["warm_identity"] = warm_identity
    evidence["cold_warm_identity_match"] = True


def prove_model_dir_variants(
    tmp_path: Path,
    artifact_root: Path,
    bundle_dir: Path,
    en_wav: Path,
    evidence: dict[str, JsonValue],
) -> None:
    results: dict[str, JsonValue] = {}
    for case in ("complete", *REQUIRED_FILES):
        poisoned = poisoned_env(tmp_path / f"poison-{case}")
        model_dir = bundle_dir
        if case != "complete":
            model_dir = tmp_path / f"bundle-missing-{case}"
            shutil.copytree(bundle_dir, model_dir)
            (model_dir / case).unlink()
        prefix = artifact_root / f"task-10-model-dir-{case}-trace"
        command = trace_command(
            prefix,
            (str(STTX_BIN), str(en_wav), "-d", str(tmp_path / f"model-{case}"), "--model-dir", str(model_dir)),
            network_only=False,
        )
        result = run_command(command, env=poisoned.subprocess_env(), cwd=Path.cwd(), timeout=900.0)
        assert result.returncode == (0 if case == "complete" else 1), result.to_json()
        assert_clean_trace(prefix, tuple(Path(value) for value in poisoned.values.values()))
        results[case] = result.to_json()
    evidence["commands"] = {
        **json_mapping(evidence["commands"], "commands"),
        "model_dir": results,
    }


def prove_real_transcript_shapes(
    tmp_path: Path,
    cold: CacheEnv,
    bundle_dir: Path,
    en_wav: Path,
    evidence: dict[str, JsonValue],
) -> None:
    two_utterances = write_two_utterance_wav(en_wav, tmp_path / "two-utterances.wav")
    two_vad = observe_vad_segments(_bundle(bundle_dir), two_utterances)
    assert len(two_vad) >= 2
    assert two_vad[1].start > 0
    result = run_sttx(
        two_utterances,
        tmp_path / "two-out",
        env=cold,
        model_dir=bundle_dir,
        offline=True,
    )
    assert result.returncode == 0, result.to_json()
    check = read_transcript(*stdout_paths(result.stdout))
    assert check.segment_count >= 2
    assert any(segment.start / 16_000 <= check.duration for segment in two_vad[1:])
    continuous = write_vad_positive_continuous_wav(
        _bundle(bundle_dir),
        en_wav,
        tmp_path / "continuous.wav",
    )
    hard_split = observe_hard_split(continuous)
    assert hard_split["first_chunk_exact"] is True
    assert hard_split["has_later_chunk"] is True
    long_result = run_sttx(
        continuous,
        tmp_path / "continuous-out",
        env=cold,
        model_dir=bundle_dir,
        offline=True,
        timeout=1800.0,
    )
    assert long_result.returncode == 0, long_result.to_json()
    evidence["checks"] = {
        **json_mapping(evidence["checks"], "checks"),
        "two_utterance_cli": {
            **asdict(check),
            "vad_chunks": [segment.to_json() for segment in two_vad],
            "second_vad_start_seconds": two_vad[1].start / 16_000,
        },
        "continuous_cli": asdict(read_transcript(*stdout_paths(long_result.stdout))),
        "hard_split": hard_split,
    }


def prove_silence_and_no_audio(
    tmp_path: Path,
    cold: CacheEnv,
    bundle_dir: Path,
    evidence: dict[str, JsonValue],
) -> None:
    silence_result = run_sttx(
        write_silence_wav(tmp_path / "silence.wav"),
        tmp_path / "silence-out",
        env=cold,
        model_dir=bundle_dir,
        offline=True,
    )
    assert silence_result.returncode == 0, silence_result.to_json()
    assert "no speech" in silence_result.stderr.lower()
    no_audio_result = run_sttx(
        write_no_audio_mp4(tmp_path / "no-audio.mp4"),
        tmp_path / "no-audio-out",
        env=cold,
        model_dir=bundle_dir,
        offline=True,
    )
    assert no_audio_result.returncode == 1, no_audio_result.to_json()
    evidence["commands"] = {
        **json_mapping(evidence["commands"], "commands"),
        "silence": silence_result.to_json(),
        "no_audio": no_audio_result.to_json(),
    }


def prove_trace_parser_hostile_fixtures(tmp_path: Path, evidence: dict[str, JsonValue]) -> None:
    prefix = tmp_path / "hostile.trace"
    (tmp_path / "hostile.trace.1").write_text(
        'chdir("/tmp") = 0\nconnect(3<socket:[1]>, 0x1, 16) = -1 ENETUNREACH\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        assert_clean_trace(prefix)
    poison = tmp_path / "poison"
    poison.mkdir()
    (tmp_path / "hostile.trace.1").write_text(
        f'chdir("{tmp_path}") = 0\nopenat(AT_FDCWD, "{poison.name}/x", O_RDONLY) = -1 ENOENT\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        assert_clean_trace(prefix, (poison,))
    long_abs = poison / ("x" * 40)
    (tmp_path / "hostile.trace.1").write_text(
        f'openat(AT_FDCWD, "{long_abs}", O_RDONLY) = -1 ENOENT\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        assert_clean_trace(prefix, (poison,))
    (tmp_path / "hostile.trace.1").write_text(
        f'openat(7<{poison}>, "nested", O_RDONLY) = 8<{poison}/nested>\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        assert_clean_trace(prefix, (poison,))
    (tmp_path / "hostile.trace.1").write_text(
        'openat(9, "unknown", O_RDONLY) = -1 ENOENT\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        assert_clean_trace(prefix)
    evidence["checks"] = {
        **json_mapping(evidence["checks"], "checks"),
        "trace_parser_hostile_fixtures": "pass",
    }


def prove_asr_hard_split(en_wav: Path, evidence: dict[str, JsonValue]) -> None:
    evidence["checks"] = {
        **json_mapping(evidence["checks"], "checks"),
        "hard_split_boundary_samples": {
            "required": 480000,
            "production_constant": __import__("sttx.asr", fromlist=["MAX_CHUNK_SAMPLES"]).MAX_CHUNK_SAMPLES,
            "sample_source": str(en_wav),
        },
    }


def pgrep_clean() -> dict[str, JsonValue]:
    completed = subprocess.run(
        ("pgrep", "-af", "sttx|ffmpeg|test_real_pipeline"),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return {"returncode": completed.returncode, "output": completed.stdout}


def stdout_paths(stdout: str) -> tuple[Path, Path]:
    lines = [Path(line) for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 2, stdout
    return lines[0], lines[1]


def _bundle(bundle_dir: Path) -> ModelBundle:
    return ModelBundle(
        encoder=bundle_dir / "encoder.int8.onnx",
        decoder=bundle_dir / "decoder.int8.onnx",
        joiner=bundle_dir / "joiner.int8.onnx",
        tokens=bundle_dir / "tokens.txt",
        silero=bundle_dir / "silero_vad.onnx",
    )
