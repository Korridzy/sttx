from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml

ROOT: Final = Path(__file__).resolve().parents[1]
SLOW: Final = "steps.detect.outputs.needs_qualification == 'true'"
ALWAYS: Final = f"always() && {SLOW}"
FILES: Final = (
    "candidate.json", "binding.json", "pipeline.json",
    "task-10-sttx-python-transcriber.log", "task-10-cleanup-receipt.json",
)
GATES: Final = ("qualification", "binding", "pipeline")
UPLOADS: Final = tuple(f"upload_{name}" for name in ("candidate", "binding", "pipeline", "log", "cleanup"))
YamlValue = str | list["YamlValue"] | dict[str, "YamlValue"]


def mapping(value: YamlValue) -> dict[str, YamlValue]:
    assert isinstance(value, dict)
    return value


def workflow() -> dict[str, YamlValue]:
    return mapping(yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader))


def steps(job: str = "backend-compat") -> dict[str, dict[str, YamlValue]]:
    items = mapping(mapping(workflow()["jobs"])[job])["steps"]
    assert isinstance(items, list)
    return {str(mapping(item)["id"]): mapping(item) for item in items if "id" in mapping(item)}


def shell(script: YamlValue, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    assert isinstance(script, str)
    return subprocess.run(["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", script],
                          env={**os.environ, **env}, cwd=ROOT, capture_output=True, text=True, timeout=30)


def test_offline_matrix_contract() -> None:
    jobs = mapping(workflow()["jobs"])
    test = mapping(jobs["test"])
    assert mapping(mapping(test["strategy"])["matrix"])["python-version"] == ["3.11", "3.12", "3.13"]
    assert test["runs-on"] == "ubuntu-latest"
    assert mapping(workflow()["on"]) == {"push": {"branches": ["main"]}, "pull_request": ""}
    items = test["steps"]
    assert isinstance(items, list)
    commands = [mapping(item).get("run") for item in items]
    assert commands == [None, "sudo apt-get update\nsudo apt-get install -y --no-install-recommends ffmpeg\n",
                        "pipx install poetry==2.4.1", None, None, "poetry install --no-interaction",
                        "poetry check", 'poetry run pytest -m "not integration"',
                        "poetry run python -m compileall -q src tests", "poetry build"]


def test_backend_structure() -> None:
    backend = mapping(mapping(workflow()["jobs"])["backend-compat"])
    selected = "${{ github.event.pull_request.head.sha || github.sha }}"
    assert "if" not in backend
    assert backend["timeout-minutes"] == "60"
    items = steps()
    assert list(items) == ["checkout", "fetch", "detect", "fast", "system", "python", "poetry", "dependencies",
                           *GATES, *UPLOADS, "aggregate"]
    assert mapping(items["checkout"]["with"]) == {"ref": selected, "fetch-depth": "0"}
    assert backend["env"] == {"HEAD_SHA": selected,
                              "BASE_SHA": "${{ github.event.pull_request.base.sha || github.event.before }}",
                              "EVIDENCE": "${{ github.workspace }}/backend-evidence"}
    assert items["fast"]["if"] == "steps.detect.outputs.needs_qualification == 'false'"
    for name in ("system", "python", "poetry", "dependencies", *GATES):
        assert items[name]["if"] == SLOW
    assert mapping(items["python"]["with"])["python-version"] == "3.13"
    assert items["poetry"]["run"] == "pipx install poetry==2.4.1"
    assert items["system"]["run"] == "sudo apt-get update\nsudo apt-get install -y --no-install-recommends ffmpeg strace\n"
    assert items["dependencies"]["run"] == "poetry install --no-interaction"
    for name, filename, option, output in zip(GATES, ("test_timing_qualification.py", "test_binding_contract.py", "test_real_pipeline.py"),
                                             ("qualification", "identity", "identity"), FILES[:3], strict=True):
        assert items[name]["continue-on-error"] == "true"
        assert items[name]["run"] == f'poetry run pytest tests/integration/{filename} -m integration -q --{option}-output="$EVIDENCE/{output}"'
    assert items["aggregate"]["if"] == ALWAYS
    assert items["aggregate"]["env"] == {name.upper(): f"${{{{ steps.{name}.outcome }}}}" for name in (*GATES, *UPLOADS)}
    assert all("cache" not in str(item.get("uses", "")) for item in items.values())


def test_each_required_upload_is_independent() -> None:
    items = steps()
    for name, filename in zip(UPLOADS, FILES, strict=True):
        assert items[name]["if"] == ALWAYS
        assert items[name]["uses"] == "actions/upload-artifact@v4"
        assert "continue-on-error" not in items[name]
        assert items[name]["with"] == {"name": f"backend-{name.removeprefix('upload_')}",
                                       "path": f"${{{{ env.EVIDENCE }}}}/{filename}", "if-no-files-found": "error"}


@pytest.mark.parametrize("value", ["true", "false", '"false"', "0", "null", "{}"])
@pytest.mark.parametrize("status", [0, 1, 2])
def test_decision_shell_accepts_only_successful_boolean(tmp_path: Path, value: str, status: int) -> None:
    output = tmp_path / "output"
    prefix = 'python3() { printf \'%s\\n\' "$*" >&2; printf \'%s\\n\' "$PAYLOAD"; return "$STATUS"; };\n'
    script = steps()["detect"]["run"]
    assert isinstance(script, str)
    result = shell(prefix + script, {"PAYLOAD": f'{{"needs_qualification":{value}}}', "STATUS": str(status),
                                    "BASE_SHA": "base", "HEAD_SHA": "head", "GITHUB_WORKSPACE": str(ROOT),
                                    "GITHUB_OUTPUT": str(output)})
    accepted = status == 0 and value in {"true", "false"}
    assert f"-I -S -B scripts/backend_change_detector.py --base base --head head --root {ROOT}" in result.stderr
    assert (result.returncode == 0) == accepted
    assert (output.read_text() if output.exists() else "") == (f"needs_qualification={value}\n" if accepted else "")


@pytest.mark.parametrize("failed", ["", *GATES, *UPLOADS, "fingerprint"])
@pytest.mark.parametrize("outcome", ["failure", "skipped", "cancelled", ""])
def test_aggregate_shell_fails_closed(failed: str, outcome: str) -> None:
    env = {name.upper(): "success" for name in (*GATES, *UPLOADS)}
    if failed:
        env[failed.upper()] = outcome
    script = steps()["aggregate"]["run"]
    assert isinstance(script, str)
    result = shell(f"poetry() {{ return {int(failed == 'fingerprint')}; }};\n" + script, env)
    assert (result.returncode == 0) == (failed == "")
    assert "scripts/compute_backend_fingerprint.py --check tests/integration/qualified_backend.json" in script


@pytest.mark.parametrize("job", ["TEST", "BACKEND"])
@pytest.mark.parametrize("outcome", ["success", "failure", "skipped", "cancelled", ""])
def test_stable_checks_shell(job: str, outcome: str) -> None:
    checks = mapping(mapping(workflow()["jobs"])["checks"])
    assert checks["needs"] == ["test", "backend-compat"]
    assert checks["if"] == "always()"
    item = steps("checks")["required"]
    assert item["env"] == {"TEST": "${{ needs.test.result }}", "BACKEND": "${{ needs.backend-compat.result }}"}
    result = shell(item["run"], {"TEST": "success", "BACKEND": "success", job: outcome})
    assert (result.returncode == 0) == (outcome == "success")


@pytest.mark.parametrize("base", ["", "0" * 40, "b" * 40])
def test_fetch_shell_preserves_missing_base(base: str) -> None:
    script = steps()["fetch"]["run"]
    assert isinstance(script, str)
    result = shell('git() { printf "%s\\n" "$*"; [[ -z "$BASE_SHA" || "$*" != *"$BASE_SHA"* ]]; };\n' + script,
                   {"BASE_SHA": base, "HEAD_SHA": "a" * 40})
    assert result.returncode == 0
    assert "fetch --no-tags origin " + "a" * 40 in result.stdout
    fetches = [line for line in result.stdout.splitlines() if line.startswith("fetch ")]
    assert len(fetches) == (2 if base and set(base) != {"0"} else 1)


def test_fast_shell_succeeds_without_artifacts(tmp_path: Path) -> None:
    result = shell(steps()["fast"]["run"], {"EVIDENCE": str(tmp_path / "evidence")})
    assert result.returncode == 0
    assert result.stdout.strip()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("gate", GATES)
def test_gate_shell_preserves_explicit_output_argument(tmp_path: Path, gate: str) -> None:
    script = steps()[gate]["run"]
    assert isinstance(script, str)
    result = shell('poetry() { printf "%s\\n" "$@"; };\n' + script, {"EVIDENCE": str(tmp_path / "with spaces")})
    arguments = result.stdout.splitlines()
    filename, option, output = {
        "qualification": ("test_timing_qualification", "qualification", "candidate"),
        "binding": ("test_binding_contract", "identity", "binding"),
        "pipeline": ("test_real_pipeline", "identity", "pipeline"),
    }[gate]
    assert result.returncode == 0
    assert arguments[:3] == ["run", "pytest", f"tests/integration/{filename}.py"]
    assert arguments[3:6] == ["-m", "integration", "-q"]
    assert arguments[6:] == [f"--{option}-output={tmp_path}/with spaces/{output}.json"]
