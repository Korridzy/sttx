from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from .release_workflow_helpers import ReleaseSandbox, release_sandbox
from .test_backend_workflows import mapping, steps, workflow
from .test_release_launcher import START

DISPATCH: Final = (steps("dispatch-release", "start-release")["dispatch"]["run"],)


@pytest.fixture
def sandbox(tmp_path: Path) -> ReleaseSandbox:
    base = release_sandbox(tmp_path)
    return replace(base, environment={
        **base.environment, "RELEASE_TAG": "v0.2.0", "RELEASE_SHA": base.environment["EXPECTED_SHA"]})


def test_dispatch_requires_successful_tag_creation_and_only_actions_write() -> None:
    job = mapping(mapping(workflow("start-release")["jobs"])["dispatch-release"])
    assert job["needs"] == ["start-release", "tag"]
    assert job["permissions"] == {"actions": "write"}
    assert "if" not in job and "continue-on-error" not in job
    action = steps("dispatch-release", "start-release")["dispatch"]
    assert action["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "RELEASE_SHA": "${{ needs.start-release.outputs.sha }}",
        "RELEASE_TAG": "${{ needs.start-release.outputs.tag }}",
    }


def test_dispatch_uses_the_validated_tag_without_creating_another_ref(sandbox: ReleaseSandbox) -> None:
    result = sandbox.run(DISPATCH)

    assert result.returncode == 0, result.stderr
    assert sandbox.requests() == [
        "api --method POST repos/Korridzy/sttx/actions/workflows/release.yml/dispatches -f ref=refs/tags/v0.2.0"]


def test_dispatch_failure_is_reported(sandbox: ReleaseSandbox) -> None:
    failed = replace(sandbox, environment={**sandbox.environment, "FAIL_REQUEST": "/dispatches"})
    summary = Path(sandbox.environment["GITHUB_STEP_SUMMARY"])
    _ = summary.write_text("before\n", encoding="utf-8")

    result = failed.run(DISPATCH)

    assert result.returncode != 0
    assert len(sandbox.requests()) == 1
    assert summary.read_text(encoding="utf-8") == "before\n"


@pytest.mark.parametrize(("job", "step"), [("tag", "tag"), ("dispatch-release", "dispatch")])
@pytest.mark.parametrize(("actor", "triggering_actor"), [("99", "Korridzy"), ("7", "other")])
def test_privileged_jobs_reject_nonowner_runs_and_reruns(
    sandbox: ReleaseSandbox, job: str, step: str, actor: str, triggering_actor: str,
) -> None:
    scenario = replace(sandbox, environment={
        **sandbox.environment, "GITHUB_ACTOR_ID": actor, "GITHUB_TRIGGERING_ACTOR": triggering_actor})

    result = scenario.run((steps(job, "start-release")[step]["run"],))

    assert result.returncode != 0
    assert sandbox.requests() == []


@pytest.mark.parametrize("name", ["release-please", "start-release", "release"])
def test_release_workflows_have_no_custom_secret_context(name: str) -> None:
    assert "secrets." not in str(workflow(name))


def test_failed_tag_creation_never_dispatches_publication(sandbox: ReleaseSandbox) -> None:
    failed = replace(sandbox, environment={**sandbox.environment, "FAIL_REQUEST": "/git/refs"})

    result = failed.run((*START, *DISPATCH))

    assert result.returncode != 0
    assert not any("/dispatches" in request for request in sandbox.requests())


def test_retrying_failed_dispatch_does_not_recreate_the_tag(sandbox: ReleaseSandbox) -> None:
    # Given an immutable tag and a dispatch that failed before a run was created.
    assert sandbox.run(START).returncode == 0
    failed = replace(sandbox, environment={**sandbox.environment, "FAIL_REQUEST": "/dispatches"})
    assert failed.run(DISPATCH).returncode != 0

    result = sandbox.run(DISPATCH)

    assert result.returncode == 0, result.stderr
    assert sum("/git/refs" in request for request in sandbox.requests()) == 1
    assert sum("/dispatches" in request for request in sandbox.requests()) == 2
