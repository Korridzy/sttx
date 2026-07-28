# noqa: SIZE_OK — Todo 10 is one external integration gate with split helpers
from __future__ import annotations

import shutil
import wave
from dataclasses import asdict
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from real_pipeline_artifacts import (
    JsonValue,
    append_log,
    bundle_identity,
    copy_bundle,
    environment_identity,
    sha256,
    snapshot_commit,
    task_artifacts,
    write_hash_manifest,
    write_json,
)
from real_pipeline_checks import (
    pgrep_clean,
    prove_asr_hard_split,
    prove_model_dir_variants,
    prove_real_transcript_shapes,
    prove_silence_and_no_audio,
    prove_trace_parser_hostile_fixtures,
    prove_warm_offline,
    stdout_paths,
)
from real_pipeline_media import (
    convert_media,
    read_transcript,
)
from real_pipeline_evidence_contract import assert_todo10_contract
from real_pipeline_observability import write_vad_positive_continuous_wav
from real_pipeline_runner import STTX_BIN, run_sttx, sanitized_env
from real_pipeline_signals import prove_signal_barriers


@pytest.mark.integration
def test_real_floating_pipeline_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_output: Path | None,
) -> None:
    artifacts = task_artifacts(identity_output)
    append_log(artifacts.log, "WORKING: Todo 10 real pipeline - cold acquisition")
    cold = sanitized_env(tmp_path / "cold")
    _apply_env(monkeypatch, cold.values)
    from sttx.model import PARAKEET_REPO_ID, SILERO_URL, resolve_bundle

    bundle = resolve_bundle()
    commit = snapshot_commit(bundle)
    wav_cache = tmp_path / "qa-wav-cache"
    en_wav = Path(
        hf_hub_download(
            repo_id=PARAKEET_REPO_ID,
            filename="test_wavs/en.wav",
            revision=commit,
            cache_dir=wav_cache,
        )
    )
    evidence: dict[str, JsonValue] = {
        "gate": "Todo 10 real floating-model pipeline",
        "venv": ".venv",
        "sttx_bin": str(STTX_BIN),
        "cold_env": cold.json(),
        "environment": environment_identity(),
        "model": {
            "repo_id": PARAKEET_REPO_ID,
            "silero_official_url": SILERO_URL,
            "commit": commit,
            "assets": bundle_identity(bundle),
        },
        "qa_wav": {
            "runtime_asset": False,
            "repo_id": PARAKEET_REPO_ID,
            "commit": commit,
            "sample_rate": _sample_rate(en_wav),
            "path": str(en_wav),
            "size": en_wav.stat().st_size,
            "sha256": sha256(en_wav),
        },
        "commands": {},
        "checks": {},
        "adversarial_classes": _adversarial_skeleton(),
    }

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    format_results: dict[str, JsonValue] = {}
    append_log(artifacts.log, "WORKING: Todo 10 real pipeline - direct CLI media")
    for suffix in (".wav", ".mp3", ".ogg", ".mp4"):
        media = convert_media(en_wav, media_dir / f"sample{suffix}")
        outdir = tmp_path / f"out-{suffix[1:]}"
        result = run_sttx(media, outdir, env=cold, timeout=1200.0)
        assert result.returncode == 0, result.to_json()
        json_path, txt_path = stdout_paths(result.stdout)
        check = read_transcript(json_path, txt_path)
        assert check.text.strip(), result.to_json()
        assert check.monotonic
        assert check.json_txt_equal
        format_results[suffix] = {
            "command": result.to_json(),
            "json": str(json_path),
            "txt": str(txt_path),
            "check": asdict(check),
        }
    evidence["commands"] = {**evidence["commands"], "direct_cli": format_results}

    bundle_dir = artifacts.root / "task-10-flat-bundle"
    evidence["flat_bundle"] = copy_bundle(bundle, bundle_dir)
    prove_warm_offline(tmp_path, artifacts.root, cold, en_wav, evidence)
    prove_model_dir_variants(tmp_path, artifacts.root, bundle_dir, en_wav, evidence)
    prove_real_transcript_shapes(tmp_path, cold, bundle_dir, en_wav, evidence)
    prove_silence_and_no_audio(tmp_path, cold, bundle_dir, evidence)
    prove_trace_parser_hostile_fixtures(tmp_path, evidence)
    prove_asr_hard_split(en_wav, evidence)
    signal_media = write_vad_positive_continuous_wav(
        bundle,
        en_wav,
        tmp_path / "signal-continuous.wav",
    )
    prove_signal_barriers(tmp_path, cold, bundle_dir, signal_media, evidence)

    shutil.rmtree(wav_cache, ignore_errors=True)
    cleanup = {
        "qa_wav_cache_removed": not wav_cache.exists(),
        "qa_en_wav_deleted": not en_wav.exists(),
        "live_process_probe": {
            **pgrep_clean(),
            "phase": "in_pytest_non_final",
            "requires_post_pytest_refresh": True,
        },
        "preserved": {
            "cold_root": str(cold.root),
            "bundle_dir": str(bundle_dir),
            "evidence_root": str(artifacts.root),
        },
    }
    evidence["cleanup"] = cleanup
    assert_todo10_contract(evidence)
    write_json(artifacts.cleanup, cleanup)
    write_json(artifacts.identity, evidence)
    write_hash_manifest(artifacts)
    write_json(
        artifacts.done_claim,
        {
            "status": "UNCOMMITTED",
            "changed_files": [
                "tests/integration/test_real_pipeline.py",
                "tests/integration/real_pipeline_artifacts.py",
                "tests/integration/real_pipeline_media.py",
                "tests/integration/real_pipeline_runner.py",
                "tests/integration/real_pipeline_trace.py",
            ],
            "evidence": str(artifacts.identity),
            "cleanup": str(artifacts.cleanup),
        },
    )


def _apply_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for key in ("HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_ASSETS_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as stream:
        return stream.getframerate()


def _adversarial_skeleton() -> dict[str, JsonValue]:
    return {
        "malformed_input": "missing bundle files, poisoned sentinels, trace fixtures, silence/no-audio",
        "prompt_injection": "N/A: no instruction-taking surface",
        "cancel_resume": "HF, Silero, and native decode SIGINT/SIGTERM barriers are probed by Todo 10",
        "stale_state": "sanitized cold root, warm offline root identity, poisoned sentinel envs, post-signal manifest equality",
        "dirty_worktree": "preserve .serena and ignored caches; commit only Todo 10 test files",
        "hung_or_long_commands": "subprocess timeouts on all CLI/trace commands",
        "flaky_tests": "deterministic media fixtures and parser fixtures",
        "misleading_success_output": "exit/XML/traces/transcript equality cross-checked",
        "repeated_interruptions": "HF, Silero, and native decode SIGINT/SIGTERM use distinct roots",
    }
