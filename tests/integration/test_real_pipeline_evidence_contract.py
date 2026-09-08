from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections.abc import Mapping

import pytest

from .real_pipeline_artifacts import JsonValue, json_mapping
from .real_pipeline_evidence_contract import (
    EvidenceContractError,
    assert_todo10_contract,
)


@pytest.mark.parametrize("after_audio", [False, True])
@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failed_pipeline_lifecycle_preserves_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_audio: bool,
    exception_type: type[BaseException],
) -> None:
    from sttx import model
    from sttx.backend_contract import PARAKEET_REVISION
    from ..model_helpers import ALL_NAMES, write_files
    from . import test_real_pipeline as pipeline

    failure = exception_type("injected lifecycle failure")
    identity = tmp_path / "evidence" / "pipeline.json"
    identity.parent.mkdir()
    _ = identity.write_text('{"gate":{"verdict":"pass","assertion":"stale"}}')
    _ = (identity.parent / "task-10-sttx-python-transcriber.log").write_text("FAILED: stale\n")
    _ = (identity.parent / "task-10-done-claim.json").write_text('{"status":"stale pass"}')
    audio = tmp_path / "qa-wav-cache" / "en.wav"
    snapshot = tmp_path / "snapshots" / PARAKEET_REVISION
    write_files(snapshot, ALL_NAMES)
    bundle = model.resolve_bundle(snapshot)

    def acquire() -> model.ModelBundle:
        started = _read_evidence(identity)
        assert started["gate"] == {"verdict": "started", "assertion": None}
        if not after_audio:
            raise failure
        return bundle

    def acquire_audio(**_kwargs: JsonValue) -> str:
        audio.parent.mkdir()
        _ = audio.write_bytes(b"owned audio")
        return str(audio)

    def fail_sample_rate(_path: Path) -> int:
        raise failure

    monkeypatch.setattr(model, "resolve_bundle", acquire)
    monkeypatch.setattr(pipeline, "hf_hub_download", acquire_audio)
    monkeypatch.setattr(pipeline, "environment_identity", lambda: {"probe": "offline"})
    monkeypatch.setattr(pipeline, "_sample_rate", fail_sample_rate)

    with pytest.raises(exception_type) as captured:
        pipeline.test_real_pinned_pipeline_gate(tmp_path, monkeypatch, identity)

    assert captured.value is failure
    evidence = _read_evidence(identity)
    assert evidence["gate"] == {
        "verdict": "fail",
        "assertion": f"{exception_type.__name__}: injected lifecycle failure",
    }
    assert evidence["failure"] == {
        "type": exception_type.__name__, "message": "injected lifecycle failure",
    }
    assert evidence["environment"] == {"probe": "offline"}
    from scripts.compute_backend_fingerprint import fingerprint

    assert evidence["venv"] == sys.prefix
    assert evidence["executable"] == sys.executable
    assert evidence["fingerprint"] == fingerprint(require_sources=True)
    assert evidence["commands"] == {}
    if after_audio:
        assert json_mapping(evidence["model"], "model")["commit"] == PARAKEET_REVISION
    cleanup = _read_evidence(identity.parent / "task-10-cleanup-receipt.json")
    assert cleanup == evidence["cleanup"]
    assert cleanup["qa_wav_cache_removed"] is True
    assert cleanup["qa_en_wav_deleted"] is True
    assert not audio.exists()
    assert not audio.parent.exists()
    lines = (identity.parent / "task-10-sttx-python-transcriber.log").read_text().splitlines()
    assert sum("FAILED:" in line for line in lines) == 1
    assert not list(identity.parent.glob("*.tmp"))
    assert not (identity.parent / "task-10-done-claim.json").exists()


@pytest.mark.parametrize("target", ["pipeline.json", "task-10-cleanup-receipt.json"])
@pytest.mark.parametrize("secondary_type", [OSError, KeyboardInterrupt])
def test_pipeline_writer_failure_preserves_original_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    secondary_type: type[BaseException],
) -> None:
    from sttx import model
    from . import test_real_pipeline as pipeline

    failure = RuntimeError("original acquisition error")
    identity = tmp_path / "evidence" / "pipeline.json"
    replace = os.replace

    def broken_replace(source: Path, destination: Path) -> None:
        if destination.name == target:
            raise secondary_type("secondary writer error")
        replace(source, destination)

    def acquire() -> model.ModelBundle:
        monkeypatch.setattr(os, "replace", broken_replace)
        raise failure

    monkeypatch.setattr(model, "resolve_bundle", acquire)
    monkeypatch.setattr(pipeline, "environment_identity", lambda: {"probe": "offline"})
    with pytest.raises(RuntimeError) as captured:
        pipeline.test_real_pinned_pipeline_gate(tmp_path, monkeypatch, identity)

    assert captured.value is failure
    assert any("secondary writer error" in note for note in failure.__notes__)
    assert not list(identity.parent.glob("*.tmp"))
    evidence = _read_evidence(identity)
    if target == "pipeline.json":
        assert json_mapping(evidence["gate"], "gate")["verdict"] == "started"
        receipt = _read_evidence(identity.parent / "task-10-cleanup-receipt.json")
        assert receipt["qa_wav_cache_removed"] is True
    else:
        assert json_mapping(evidence["gate"], "gate")["verdict"] == "fail"
        assert json_mapping(evidence["failure"], "failure")["message"] == "original acquisition error"


def _read_evidence(path: Path) -> Mapping[str, JsonValue]:
    payload: JsonValue = json.loads(path.read_text())
    return json_mapping(payload, "evidence")

def test_current_false_positive_evidence_is_rejected_when_present(tmp_path: Path) -> None:
    payload = _valid_evidence(tmp_path)
    model = payload["model"]
    assert isinstance(model, dict)
    assets = model["assets"]
    assert isinstance(assets, dict)
    encoder = assets["encoder"]
    assert isinstance(encoder, dict)
    encoder["path"] = str(tmp_path / "outside-cache" / "encoder.int8.onnx")
    with pytest.raises(
        EvidenceContractError,
        match="outside isolated HF cache",
    ):
        assert_todo10_contract(payload)


def test_warm_identity_and_metadata_are_required(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    evidence.pop("warm_identity")
    with pytest.raises(EvidenceContractError, match="warm_identity"):
        assert_todo10_contract(evidence)

    evidence = _valid_evidence(tmp_path)
    warm = evidence["warm_identity"]
    assert isinstance(warm, dict)
    warm["commit"] = "different"
    with pytest.raises(EvidenceContractError, match="warm commit"):
        assert_todo10_contract(evidence)

    for field in ("path", "resolved_path", "size", "sha256"):
        evidence = _valid_evidence(tmp_path)
        warm = evidence["warm_identity"]
        assert isinstance(warm, dict)
        assets = warm["assets"]
        assert isinstance(assets, dict)
        encoder = assets["encoder"]
        assert isinstance(encoder, dict)
        encoder[field] = "different" if field != "size" else 1
        with pytest.raises(EvidenceContractError, match=rf"warm encoder\.{field}"):
            assert_todo10_contract(evidence)

    evidence = _valid_evidence(tmp_path)
    evidence["cold_warm_identity_match"] = False
    with pytest.raises(EvidenceContractError, match="comparison"):
        assert_todo10_contract(evidence)

    evidence = _valid_evidence(tmp_path)
    model = evidence["model"]
    assert isinstance(model, dict)
    model.pop("repo_id")
    with pytest.raises(EvidenceContractError, match="repo_id"):
        assert_todo10_contract(evidence)

    evidence = _valid_evidence(tmp_path)
    qa_wav = evidence["qa_wav"]
    assert isinstance(qa_wav, dict)
    qa_wav.pop("sample_rate")
    with pytest.raises(EvidenceContractError, match="sample_rate"):
        assert_todo10_contract(evidence)


def test_qa_wav_source_sample_rate_accepts_observed_upstream_rate(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    qa_wav = evidence["qa_wav"]
    assert isinstance(qa_wav, dict)
    qa_wav["sample_rate"] = 24000
    assert_todo10_contract(evidence)

    for invalid_rate in (0, -24000, "24000"):
        evidence = _valid_evidence(tmp_path)
        qa_wav = evidence["qa_wav"]
        assert isinstance(qa_wav, dict)
        qa_wav["sample_rate"] = invalid_rate
        with pytest.raises(EvidenceContractError, match="sample_rate|positive integer"):
            assert_todo10_contract(evidence)


def test_signal_barriers_reject_bookkeeping_and_cpu_only_proofs(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    hf_probe = _first_probe(evidence, "hf", "SIGINT")
    for partial in (
        tmp_path / "cold" / "hf" / "xet" / "logs" / "xet.log",
        tmp_path / "cold" / "hf" / "xet" / "CACHEDIR.TAG",
        tmp_path / "cold" / "hf" / "xet" / "metadata" / "tree.json",
        tmp_path / "cold" / "hf" / "hub" / "refs" / "main",
    ):
        hf_probe["barrier"] = {
            "partial_asset_candidates": [str(partial)],
            "partial_asset_kinds": {str(partial): "xet_data_partial"},
            "partial_asset_sizes": {str(partial): 40},
            "process_live_at_barrier": True,
            "runtime_complete_count": 0,
            "missing_runtime_finals": ["encoder.int8.onnx"],
        }
        with pytest.raises(EvidenceContractError, match="bookkeeping"):
            assert_todo10_contract(evidence)

    evidence = _valid_evidence(tmp_path)
    hf_probe = _first_probe(evidence, "hf", "SIGINT")
    partial = tmp_path / "cold" / "hf" / "xet" / "chunk-cache" / "runtime.chunk"
    hf_probe["barrier"] = {
        "partial_asset_candidates": [str(partial)],
        "partial_asset_kinds": {str(partial): "xet_data_partial"},
        "partial_asset_sizes": {str(partial): 40},
        "process_live_at_barrier": True,
        "runtime_complete_count": 0,
        "missing_runtime_finals": ["encoder.int8.onnx"],
    }
    assert_todo10_contract(evidence)

    evidence = _valid_evidence(tmp_path)
    silero_probe = _first_probe(evidence, "silero", "SIGINT")
    silero_probe["barrier"] = {
        "ready": str(tmp_path / ".silero_vad.onnx.tmp"),
        "staged_path": str(tmp_path / ".silero_vad.onnx.tmp"),
        "official_url": "https://example.invalid/silero_vad.onnx",
    }
    with pytest.raises(EvidenceContractError, match="staged_size"):
        assert_todo10_contract(evidence)

    evidence = _valid_evidence(tmp_path)
    decode_probe = _first_probe(evidence, "native_decode", "SIGTERM")
    decode_probe["barrier"] = {
        "cpu_ticks_before": 10,
        "cpu_ticks_after": 11,
        "ready": str(tmp_path / "ready"),
    }
    with pytest.raises(EvidenceContractError, match="real decode marker"):
        assert_todo10_contract(evidence)


def test_cleanup_probe_inside_pytest_must_be_marked_non_final(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    cleanup = evidence["cleanup"]
    assert isinstance(cleanup, dict)
    cleanup["live_process_probe"] = {
        "phase": "final",
        "requires_post_pytest_refresh": False,
        "returncode": 0,
        "output": "123 pytest -q -m integration tests/integration/test_real_pipeline.py",
    }
    with pytest.raises(EvidenceContractError, match="non-final marker"):
        assert_todo10_contract(evidence)


def _valid_evidence(tmp_path: Path) -> dict[str, JsonValue]:
    repo_id = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    commit = "commit"
    hf_hub = tmp_path / "cold" / "hf" / "hub"
    snapshot = hf_hub / "models--repo" / "snapshots" / commit
    blob = hf_hub / "models--repo" / "blobs" / "hash"
    assets = {
        name: {
            "path": str(snapshot / filename),
            "resolved_path": str(blob / filename),
            "size": 64,
            "sha256": f"{name}-sha256",
        }
        for name, filename in {
            "encoder": "encoder.int8.onnx",
            "decoder": "decoder.int8.onnx",
            "joiner": "joiner.int8.onnx",
            "tokens": "tokens.txt",
        }.items()
    }
    assets["silero"] = {
        "path": str(tmp_path / "cold" / "home" / ".cache" / "sttx" / "silero_vad.onnx"),
        "resolved_path": str(tmp_path / "cold" / "home" / ".cache" / "sttx" / "silero_vad.onnx"),
        "size": 64,
        "sha256": "silero-sha256",
    }
    warm_assets = {name: dict(identity) for name, identity in assets.items()}
    probes: list[JsonValue] = []
    for phase in ("hf", "silero", "native_decode"):
        for signum in ("SIGINT", "SIGTERM"):
            probes.append({"phase": phase, "signum": signum, "barrier": _barrier(tmp_path, phase)})
    return {
        "cold_env": {"values": {"HF_HUB_CACHE": str(hf_hub)}},
        "model": {
            "repo_id": repo_id,
            "silero_official_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
            "commit": commit,
            "assets": assets,
        },
        "qa_wav": {
            "runtime_asset": False,
            "repo_id": repo_id,
            "commit": commit,
            "sample_rate": 24000,
            "path": str(tmp_path / "en.wav"),
            "size": 64,
            "sha256": "qa-wav-sha256",
        },
        "warm_identity": {"repo_id": repo_id, "commit": commit, "assets": warm_assets},
        "cold_warm_identity_match": True,
        "signals": {"probes": probes},
        "cleanup": {
            "live_process_probe": {
                "phase": "in_pytest_non_final",
                "requires_post_pytest_refresh": True,
                "output": "pytest tests/integration/test_real_pipeline.py",
            }
        },
    }


def _barrier(tmp_path: Path, phase: str) -> dict[str, JsonValue]:
    match phase:
        case "hf":
            partial = tmp_path / "cold" / "hf" / "hub" / "blobs" / "encoder.incomplete"
            return {
                "partial_asset_candidates": [str(partial)],
                "partial_asset_kinds": {str(partial): "hub_incomplete"},
                "partial_asset_sizes": {str(partial): 64},
                "process_live_at_barrier": True,
                "runtime_complete_count": 0,
                "missing_runtime_finals": ["encoder.int8.onnx"],
            }
        case "silero":
            staged = str(tmp_path / ".silero_vad.onnx.tmp")
            return {"ready": staged, "staged_path": staged, "staged_size": 64}
        case "native_decode":
            return {
                "cpu_ticks_before": 10,
                "cpu_ticks_after": 11,
                "real_decode_entered": str(tmp_path / "real_decode_entered"),
                "output_finals_exist": False,
            }
        case _:
            raise AssertionError(f"unknown phase {phase}")


def _first_probe(evidence: dict[str, JsonValue], phase: str, signum: str) -> dict[str, JsonValue]:
    signals = evidence["signals"]
    assert isinstance(signals, dict)
    probes = signals["probes"]
    assert isinstance(probes, list)
    for probe in probes:
        assert isinstance(probe, dict)
        if probe["phase"] == phase and probe["signum"] == signum:
            return probe
    raise AssertionError(f"missing probe {phase}/{signum}")
