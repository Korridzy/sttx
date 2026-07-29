from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Final, TypedDict

import pytest

from signal_driver import PHASES

DRIVER: Final = Path(__file__).with_name("signal_driver.py")
DEADLINE_SECONDS: Final = 5.0


class SignalResult(TypedDict):
    phase: str
    exit_code: int
    process_exit: int
    handlers_restored: bool
    json: str
    txt: str
    staging: list[str]
    prepared_exists: bool
    hf_partial_valid: bool
    stderr: str
    stdout: str
    elapsed: float
    ffmpeg_pid: int | None


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"driver exited before ready: {process.returncode=} {stdout=} {stderr=}"
            )
        time.sleep(0.01)
    process.kill()
    process.wait()
    pytest.fail(f"driver did not become ready: {path}")


def _release_cleanup(root: Path) -> None:
    try:
        descriptor = os.open(root / "cleanup-release", os.O_WRONLY | os.O_NONBLOCK)
    except OSError as error:
        if error.errno == errno.ENXIO:
            return
        raise
    with os.fdopen(descriptor, "wb") as release:
        release.write(b"x")


def _run_signal(
    tmp_path: Path,
    phase: str,
    signum: signal.Signals,
    *,
    repeated: bool | signal.Signals = False,
    run_index: int = 0,
) -> SignalResult:
    root = tmp_path / f"{phase}-{signum.name}-{run_index}"
    process = subprocess.Popen(
        [sys.executable, str(DRIVER), phase, str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for(root / "ready", process)
    started = time.monotonic()
    process.send_signal(signum)
    match repeated:
        case signal.Signals() as later:
            _wait_for(root / "cleanup-ready", process)
            process.send_signal(later)
            _release_cleanup(root)
        case True:
            process.send_signal(signum)
        case False:
            pass
    stdout, stderr = process.communicate(timeout=DEADLINE_SECONDS)
    elapsed = time.monotonic() - started
    result_path = root / "result.json"
    if not result_path.exists():
        pid_path = root / "ffmpeg.pid"
        if pid_path.exists():
            ffmpeg_pid = int(pid_path.read_text(encoding="utf-8"))
            try:
                os.killpg(ffmpeg_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        pytest.fail(
            f"driver exited without result: {process.returncode=} {stdout=} {stderr=}"
        )
    result: SignalResult = json.loads(
        result_path.read_text(encoding="utf-8")
    )
    pid_path = root / "ffmpeg.pid"
    result.update(
        {
            "stdout": stdout,
            "stderr": stderr,
            "elapsed": elapsed,
            "process_exit": process.returncode,
            "ffmpeg_pid": int(pid_path.read_text(encoding="utf-8"))
            if pid_path.exists()
            else None,
        }
    )
    return result


def _assert_cancelled(result: SignalResult, signum: signal.Signals) -> None:
    assert result["exit_code"] == 128 + signum
    assert result["process_exit"] == 128 + signum
    assert result["elapsed"] < DEADLINE_SECONDS
    assert result["stdout"] == ""
    assert "cancel" in result["stderr"].lower()
    assert "Traceback" not in result["stderr"]
    assert result["staging"] == []
    assert result["prepared_exists"] is False
    assert result["handlers_restored"] is True
    assert result["hf_partial_valid"] is False
    if result["ffmpeg_pid"] is not None:
        with pytest.raises(ProcessLookupError):
            os.kill(result["ffmpeg_pid"], 0)


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("signum", (signal.SIGINT, signal.SIGTERM))
def test_signal_phase_matrix_exits_and_cleans(
    tmp_path: Path,
    phase: str,
    signum: signal.Signals,
) -> None:
    first = _run_signal(tmp_path, phase, signum)
    second = _run_signal(
        tmp_path,
        phase,
        signum,
        repeated=phase == "decode",
        run_index=1,
    )
    _assert_cancelled(first, signum)
    _assert_cancelled(second, signum)
    comparable = (
        "exit_code",
        "process_exit",
        "handlers_restored",
        "json",
        "txt",
        "staging",
        "prepared_exists",
        "hf_partial_valid",
    )
    assert {key: first[key] for key in comparable} == {
        key: second[key] for key in comparable
    }


@pytest.mark.parametrize("signum", (signal.SIGINT, signal.SIGTERM))
def test_signal_before_first_replace_preserves_finals(
    tmp_path: Path,
    signum: signal.Signals,
) -> None:
    result = _run_signal(tmp_path, "before_replace", signum)
    _assert_cancelled(result, signum)
    assert result["json"] == "old-json"
    assert result["txt"] == "old-txt"


@pytest.mark.parametrize("signum", (signal.SIGINT, signal.SIGTERM))
def test_signal_between_replaces_allows_only_documented_mixed_pair(
    tmp_path: Path,
    signum: signal.Signals,
) -> None:
    result = _run_signal(tmp_path, "between_replaces", signum)
    _assert_cancelled(result, signum)
    assert json.loads(result["json"])["text"] == "new text"
    assert result["txt"] == "old-txt"


@pytest.mark.parametrize("signum", (signal.SIGINT, signal.SIGTERM))
def test_handlers_are_restored(
    tmp_path: Path,
    signum: signal.Signals,
) -> None:
    result = _run_signal(tmp_path, "decode", signum)
    _assert_cancelled(result, signum)


@pytest.mark.parametrize(
    ("first", "later"),
    (
        (signal.SIGINT, signal.SIGTERM),
        (signal.SIGTERM, signal.SIGINT),
    ),
)
def test_first_signal_wins_when_later_signal_arrives_during_cleanup(
    tmp_path: Path,
    first: signal.Signals,
    later: signal.Signals,
) -> None:
    result = _run_signal(tmp_path, "cleanup", first, repeated=later)
    _assert_cancelled(result, first)
    assert f"cancelled by {first.name}" in result["stderr"]
    assert f"cancelled by {later.name}" not in result["stderr"]
    assert result["json"] == "old-json"
    assert result["txt"] == "old-txt"
