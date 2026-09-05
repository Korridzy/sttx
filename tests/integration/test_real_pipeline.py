# noqa: SIZE_OK — Todo 10 is one external integration gate with split helpers
from __future__ import annotations

import json
import shutil
import wave
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from .real_pipeline_artifacts import (
    JsonValue,
    TaskArtifacts,
    append_log,
    bundle_identity,
    copy_bundle,
    environment_identity,
    json_mapping,
    sha256,
    snapshot_commit,
    task_artifacts,
    write_hash_manifest,
    write_json,
)
from .real_pipeline_checks import (
    pgrep_clean,
    prove_asr_hard_split,
    prove_model_dir_variants,
    prove_real_transcript_shapes,
    prove_silence_and_no_audio,
    prove_trace_parser_hostile_fixtures,
    prove_warm_offline,
    stdout_paths,
)
from .real_pipeline_media import (
    convert_media,
    read_transcript,
)
from .real_pipeline_evidence_contract import assert_todo10_contract
from .real_pipeline_observability import write_vad_positive_continuous_wav
from .real_pipeline_runner import STTX_BIN, CacheEnv, run_sttx, sanitized_env
from .real_pipeline_signals import prove_signal_barriers


@pytest.mark.integration
def test_real_pinned_pipeline_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_output: Path | None,
) -> None:
    artifacts = task_artifacts(identity_output, tmp_path)
    evidence: dict[str, JsonValue] = {
        "gate": {"verdict": "started", "assertion": None},
        "commands": {},
        "checks": {},
        "cleanup": {
            "qa_wav_cache": str(tmp_path / "qa-wav-cache"),
            "qa_en_wav": None,
        },
        "adversarial_classes": _adversarial_skeleton(),
    }
    try:
        _write_evidence(artifacts.identity, evidence)
        _ = artifacts.log.write_text("", encoding="utf-8")
        artifacts.done_claim.unlink(missing_ok=True)
        artifacts.hash_manifest.unlink(missing_ok=True)
        _write_evidence(artifacts.cleanup, json_mapping(evidence["cleanup"], "cleanup"))
        append_log(artifacts.log, "WORKING: Todo 10 real pipeline - cold acquisition")
        cold = sanitized_env(tmp_path / "cold")
        _apply_env(monkeypatch, cold.values)
        _run_pipeline(cold, artifacts, evidence)
    except BaseException as error:
        _finalize_failure(artifacts, evidence, error)
        raise


def _run_pipeline(
    cold: CacheEnv,
    artifacts: TaskArtifacts,
    evidence: dict[str, JsonValue],
) -> None:
    tmp_path = cold.root.parent
    wav_cache = tmp_path / "qa-wav-cache"
    from sttx.model import PARAKEET_REPO_ID, PARAKEET_REVISION, SILERO_URL, resolve_bundle

    evidence.update({
        "venv": ".venv",
        "sttx_bin": str(STTX_BIN),
        "cold_env": cold.json(),
        "environment": environment_identity(),
    })
    bundle = resolve_bundle()
    commit = snapshot_commit(bundle)
    evidence["model"] = {
        "repo_id": PARAKEET_REPO_ID,
        "silero_official_url": SILERO_URL,
        "commit": commit,
        "assets": bundle_identity(bundle),
    }
    assert commit == PARAKEET_REVISION, (commit, PARAKEET_REVISION)
    en_wav = Path(
        hf_hub_download(
            repo_id=PARAKEET_REPO_ID,
            filename="test_wavs/en.wav",
            revision=commit,
            cache_dir=wav_cache,
        )
    )
    evidence["cleanup"] = {"qa_wav_cache": str(wav_cache), "qa_en_wav": str(en_wav)}
    evidence["qa_wav"] = {
        "runtime_asset": False,
        "repo_id": PARAKEET_REPO_ID,
        "commit": commit,
        "sample_rate": _sample_rate(en_wav),
        "path": str(en_wav),
        "size": en_wav.stat().st_size,
        "sha256": sha256(en_wav),
    }

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    format_results: dict[str, JsonValue] = {}
    evidence["commands"] = {"direct_cli": format_results}
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
    evidence["commands"] = {
        **json_mapping(evidence["commands"], "commands"),
        "direct_cli": format_results,
    }

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
    cleanup: dict[str, JsonValue] = {
        "qa_wav_cache": str(wav_cache),
        "qa_en_wav": str(en_wav),
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
    evidence["gate"] = {"verdict": "pass", "assertion": "all"}
    _write_evidence(artifacts.cleanup, cleanup)
    _write_evidence(artifacts.identity, evidence)
    write_hash_manifest(artifacts)
    _write_evidence(
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


def _write_evidence(path: Path, payload: Mapping[str, JsonValue]) -> None:
    try:
        write_json(path, payload)
    except BaseException as error:
        try:
            path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
        except BaseException as secondary:
            error.add_note(f"staging cleanup failed: {type(secondary).__name__}: {secondary}")
        raise


def _finalize_failure(
    artifacts: TaskArtifacts,
    evidence: dict[str, JsonValue],
    error: BaseException,
) -> None:
    failure = {"type": type(error).__name__, "message": str(error)}
    evidence["gate"] = {
        "verdict": "fail", "assertion": f"{type(error).__name__}: {error}",
    }
    evidence["failure"] = failure
    for path in (artifacts.hash_manifest, artifacts.done_claim):
        try:
            path.with_name(f".{path.name}.tmp").unlink(missing_ok=True)
        except BaseException as secondary:
            error.add_note(f"staging cleanup failed: {type(secondary).__name__}: {secondary}")
    cleanup = dict(json_mapping(evidence.get("cleanup", {}), "cleanup"))
    cache = cleanup.get("qa_wav_cache")
    if isinstance(cache, str):
        try:
            shutil.rmtree(cache, ignore_errors=True)
        except BaseException as secondary:
            error.add_note(f"audio cleanup failed: {type(secondary).__name__}: {secondary}")
        cleanup["qa_wav_cache_removed"] = not Path(cache).exists()
        audio = cleanup.get("qa_en_wav")
        cleanup["qa_en_wav_deleted"] = (
            not Path(audio).exists() if isinstance(audio, str) else not Path(cache).exists()
        )
    evidence["cleanup"] = cleanup
    try:
        append_log(artifacts.log, "FAILED: " + json.dumps(failure, ensure_ascii=True))
    except BaseException as secondary:
        error.add_note(f"failure log failed: {type(secondary).__name__}: {secondary}")
    for path, payload in ((artifacts.cleanup, cleanup), (artifacts.identity, evidence)):
        try:
            _write_evidence(path, payload)
        except BaseException as secondary:
            error.add_note(f"evidence write failed: {type(secondary).__name__}: {secondary}")


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
