from __future__ import annotations

import json
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import pytest

from tests.backend_qualification_helpers import baseline_document, observation_case
from tests.integration import qualification_schema as schema
from tests.integration import test_timing_qualification as gate


def test_gate_when_acquisition_fails_preserves_original_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.integration import qualification_probes as probes
    from tests.integration.test_timing_qualification import run_qualification

    output = tmp_path / "candidate.json"
    failure = RuntimeError("acquisition sentinel")

    def fail() -> NoReturn:
        assert json.loads(output.read_text())["phase"] == "started"
        raise failure

    monkeypatch.setattr(probes, "resolve_bundle", fail)

    with pytest.raises(RuntimeError) as raised:
        run_qualification(output, tmp_path, tmp_path / "absent.json")

    assert raised.value is failure
    artifact = json.loads(output.read_text())
    assert artifact["record_kind"] == "qualification_failure_envelope"
    assert artifact["phase"] == "failed"
    assert artifact["exception"] == {"type": "RuntimeError", "message": "acquisition sentinel"}


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_gate_when_output_aliases_baseline_preserves_bytes(tmp_path: Path, alias: str) -> None:
    from tests.integration.test_timing_qualification import run_qualification

    baseline = tmp_path / "baseline.json"
    original = b"untrusted baseline bytes"
    _ = baseline.write_bytes(original)
    output = baseline if alias == "same" else tmp_path / "candidate.json"
    if alias == "symlink":
        output.symlink_to(baseline)
    if alias == "hardlink":
        output.hardlink_to(baseline)

    with pytest.raises(ValueError):
        run_qualification(output, tmp_path, baseline)

    assert baseline.read_bytes() == original


@pytest.mark.parametrize("baseline_bytes", [b"{", b"null", b"\xff", b'{"a":1,"a":2}'])
def test_gate_when_baseline_malformed_keeps_full_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        baseline_bytes: bytes) -> None:
    observations, expected = observation_case()
    baseline = tmp_path / "baseline.json"
    _ = baseline.write_bytes(baseline_bytes)
    output = tmp_path / "candidate.json"
    owned = tmp_path / "owned.wav"

    @contextmanager
    def collect(scratch: Path, settings: schema.Expectations) -> Iterator[schema.Observations]:
        from sttx.audio import PreparedAudio
        assert scratch == tmp_path and settings == expected
        with PreparedAudio(owned, 1):
            _ = owned.write_bytes(b"owned")
            yield observations

    monkeypatch.setattr(gate, "expectations", lambda: expected)
    monkeypatch.setattr(gate.probes, "collect_observations", collect)

    with pytest.raises(AssertionError):
        gate.run_qualification(output, tmp_path, baseline)

    candidate = json.loads(output.read_text())
    assert candidate["record_kind"] == "qualification_candidate"
    assert candidate["baseline_comparison"]["integrity"][0]["code"] == "baseline.invalid"
    assert baseline.read_bytes() == baseline_bytes
    assert not owned.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_gate_when_late_cleanup_exception_retains_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observations, expected = observation_case()
    baseline = tmp_path / "baseline.json"
    _ = baseline.write_text(json.dumps(baseline_document(observations)))
    output = tmp_path / "candidate.json"
    failure = RuntimeError("late cleanup sentinel")

    @contextmanager
    def collect(scratch: Path, settings: schema.Expectations) -> Iterator[schema.Observations]:
        del scratch, settings
        yield observations
        raise failure

    monkeypatch.setattr(gate, "expectations", lambda: expected)
    monkeypatch.setattr(gate.contract, "QUALIFIED_RUNS_SHA256", observations["signature_sha256"])
    monkeypatch.setattr(gate.probes, "collect_observations", collect)

    with pytest.raises(RuntimeError) as raised:
        gate.run_qualification(output, tmp_path, baseline)

    assert raised.value is failure
    assert json.loads(output.read_text())["record_kind"] == "qualification_candidate"


@pytest.mark.parametrize("failure", [RuntimeError("early"), KeyboardInterrupt("interrupted"), SystemExit(143)])
def test_gate_when_failure_after_audio_ownership_cleans_and_propagates(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> None:
    from sttx.audio import PreparedAudio

    owned = tmp_path / "owned.wav"
    output = tmp_path / "candidate.json"

    @contextmanager
    def collect(scratch: Path, settings: schema.Expectations) -> Iterator[schema.Observations]:
        del scratch, settings
        with PreparedAudio(owned, 1):
            _ = owned.write_bytes(b"owned")
            raise failure
        yield observation_case()[0]

    monkeypatch.setattr(gate.probes, "collect_observations", collect)

    with pytest.raises(type(failure)) as raised:
        gate.run_qualification(output, tmp_path, tmp_path / "absent.json")

    assert raised.value is failure
    assert not owned.exists()
    artifact = json.loads(output.read_text())
    assert artifact["exception"] == {"type": type(failure).__name__, "message": str(failure)}
    assert artifact["phase"] == "failed"


def test_gate_when_failure_artifact_write_fails_keeps_original_exception(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import JsonValue, write_json
    from collections.abc import Mapping

    output = tmp_path / "candidate.json"
    failure = RuntimeError("acquisition sentinel")

    def acquire() -> NoReturn:
        raise failure

    def write(path: Path, payload: Mapping[str, JsonValue]) -> None:
        if payload.get("phase") == "failed":
            raise OSError("artifact storage unavailable")
        write_json(path, payload)

    monkeypatch.setattr(gate.probes, "resolve_bundle", acquire)
    monkeypatch.setattr(gate, "write_json", write)

    with pytest.raises(RuntimeError) as raised:
        gate.run_qualification(output, tmp_path, tmp_path / "absent.json")

    assert raised.value is failure
    assert json.loads(output.read_text())["phase"] == "started"
    assert "artifact" in failure.__notes__[0]


def test_gate_when_candidate_valid_returns_written_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observations, expected = observation_case()
    baseline = tmp_path / "baseline.json"
    _ = baseline.write_text(json.dumps(baseline_document(observations)))

    @contextmanager
    def collect(scratch: Path, settings: schema.Expectations) -> Iterator[schema.Observations]:
        del scratch, settings
        yield observations

    monkeypatch.setattr(gate, "expectations", lambda: expected)
    monkeypatch.setattr(gate.contract, "QUALIFIED_RUNS_SHA256", observations["signature_sha256"])
    monkeypatch.setattr(gate.probes, "collect_observations", collect)

    candidate = gate.run_qualification(tmp_path / "candidate.json", tmp_path, baseline)

    assert json.loads((tmp_path / "candidate.json").read_text()) == candidate
    assert candidate["baseline_comparison"] == {"comparable": True, "integrity": [], "identity": [], "behavior": []}


def test_gate_when_started_output_is_directory_never_acquires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def acquire() -> NoReturn:
        raise AssertionError("acquisition must not start without an artifact")

    monkeypatch.setattr(gate.probes, "resolve_bundle", acquire)

    with pytest.raises(OSError):
        gate.run_qualification(tmp_path, tmp_path, tmp_path / "absent.json")

    assert list(tmp_path.iterdir()) == []


def test_writer_when_replace_fails_preserves_old_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests import conftest

    output = tmp_path / "candidate.json"
    _ = output.write_bytes(b"previous document")

    def replace(source: Path, destination: Path) -> NoReturn:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(conftest.os, "replace", replace)

    with pytest.raises(OSError):
        conftest.write_json(output, {"new": True})

    assert output.read_bytes() == b"previous document"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("real_signal", [False, True], ids=["injected-interruption", "actual-sigint"])
def test_gate_when_secondary_interruption_at_replace_preserves_original(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, real_signal: bool) -> None:
    from tests import conftest

    output = tmp_path / "candidate.json"
    original = RuntimeError("original acquisition failure")
    started: list[bytes] = []
    staging_paths: list[Path] = []

    def interrupted_replace(source: Path, destination: Path) -> None:
        assert destination == output
        assert json.loads(source.read_text())["phase"] == "failed"
        staging_paths.append(source)
        if real_signal:
            signal.raise_signal(signal.SIGINT)
        else:
            raise KeyboardInterrupt("secondary interruption")

    def acquire() -> NoReturn:
        started.append(output.read_bytes())
        monkeypatch.setattr(conftest.os, "replace", interrupted_replace)
        raise original

    monkeypatch.setattr(gate.probes, "resolve_bundle", acquire)
    previous_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(BaseException) as caught:
            gate.run_qualification(output, tmp_path, tmp_path / "absent.json")
    finally:
        _ = signal.signal(signal.SIGINT, previous_handler)

    assert caught.value is original
    assert len(original.__notes__) == 1
    assert "could not be updated" in original.__notes__[0]
    assert "KeyboardInterrupt" in original.__notes__[0]
    assert output.read_bytes() == started[0]
    assert json.loads(output.read_text())["phase"] == "started"
    assert len(staging_paths) == 1 and not staging_paths[0].exists()
    assert list(tmp_path.glob(".*.tmp")) == []
