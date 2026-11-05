from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from .release_workflow_helpers import ReleaseSandbox, release_sandbox
from .test_backend_workflows import CHECKOUT_ACTION, mapping, steps, workflow
from .test_repo_contract import git

START: Final = (
    *(steps("start-release", "start-release")[name]["run"] for name in ("release-pr", "validate")),
    steps("tag", "start-release")["tag"]["run"],
)
LABELS: Final = (steps("release-labels", "release")["labels"]["run"],)


@pytest.fixture
def sandbox(tmp_path: Path) -> ReleaseSandbox:
    return release_sandbox(tmp_path)


def test_start_release_has_only_a_main_scoped_manual_entry() -> None:
    release = workflow("start-release")
    inputs = mapping(mapping(mapping(release["on"])["workflow_dispatch"])["inputs"])
    assert set(inputs) == {"release-pr"}
    assert mapping(inputs["release-pr"])["required"] == "true"
    assert mapping(inputs["release-pr"])["type"] == "string"
    assert release["permissions"] == {"contents": "read", "pull-requests": "read", "actions": "read"}
    assert release["concurrency"] == {"group": "start-release", "cancel-in-progress": "false"}
    jobs = mapping(release["jobs"])
    assert list(jobs) == ["start-release", "tag", "dispatch-release"]
    launcher = mapping(jobs["start-release"])
    assert launcher["if"] == "github.ref == 'refs/heads/main'"
    assert "permissions" not in launcher
    assert launcher["outputs"] == {
        "sha": "${{ steps.release-pr.outputs.sha }}", "tag": "${{ steps.validate.outputs.tag }}"}
    items = steps("start-release", "start-release")
    assert list(items) == ["release-pr", "checkout", "validate"]
    assert all("continue-on-error" not in item and "if" not in item for item in items.values())
    assert items["checkout"]["uses"] == CHECKOUT_ACTION
    assert items["checkout"]["with"] == {
        "ref": "${{ steps.release-pr.outputs.sha }}", "fetch-depth": "0", "persist-credentials": "false"}
    for name in ("release-pr", "validate"):
        assert mapping(items[name]["env"])["GH_TOKEN"] == "${{ github.token }}"
    tag = mapping(jobs["tag"])
    assert tag["needs"] == ["start-release"]
    assert tag["permissions"] == {"contents": "write"}
    assert len(tag["steps"]) == 1
    assert steps("tag", "start-release")["tag"]["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "RELEASE_SHA": "${{ needs.start-release.outputs.sha }}",
        "RELEASE_TAG": "${{ needs.start-release.outputs.tag }}",
        "RELEASE_PR": "${{ inputs.release-pr }}",
    }
    assert mapping(items["release-pr"]["env"])["RELEASE_PR"] == "${{ inputs.release-pr }}"


def test_start_release_tags_exactly_the_reviewed_commit(sandbox: ReleaseSandbox) -> None:
    result = sandbox.run(START)
    assert result.returncode == 0, result.stderr
    assert sandbox.requests()[-1] == (
        "api --method POST repos/Korridzy/sttx/git/refs -f ref=refs/tags/v0.2.0 "
        f"-f sha={sandbox.environment['EXPECTED_SHA']}")
    assert (sandbox.root / "outputs").read_text().splitlines() == [
        f"sha={sandbox.environment['EXPECTED_SHA']}", "tag=v0.2.0"]


@pytest.mark.parametrize("change", [
    ".merged = false", '.base.ref = "feature"', '.base.repo.full_name = "other/repo"',
    '.head.repo.full_name = "other/repo"', '.head.ref = "feature"', '.user.type = "User"',
    '.user.login = "Korridzy"', '.user.login = "other[bot]"', ".user.login = null",
    ".labels = []", '.merge_commit_sha = "not-a-sha"',
    '.merge_commit_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
])
def test_start_release_rejects_an_unapproved_pr(sandbox: ReleaseSandbox, change: str) -> None:
    scenario = replace(sandbox, environment={**sandbox.environment, "PR_FILTER": change})
    assert scenario.run(START).returncode != 0
    assert not any("--method POST" in request for request in sandbox.requests())


@pytest.mark.parametrize("number", ["", "0", "-1", "41/commits", "41\n42", "$(touch injected)"])
def test_start_release_rejects_invalid_input_before_lookup(sandbox: ReleaseSandbox, number: str) -> None:
    scenario = replace(sandbox, environment={**sandbox.environment, "RELEASE_PR": number})
    assert scenario.run(START).returncode != 0
    assert sandbox.requests() == []


@pytest.mark.parametrize("change", [
    ".workflow_runs = []", '.workflow_runs[0].head_sha = "wrong"',
    '.workflow_runs[0].head_branch = "feature"', '.workflow_runs[0].event = "pull_request"',
    '.workflow_runs[0].status = "in_progress"', '.workflow_runs[0].conclusion = "failure"',
    '.workflow_runs[0].conclusion = "cancelled"', '.workflow_runs[0].conclusion = "skipped"',
])
def test_start_release_requires_successful_ci_on_the_exact_main_commit(sandbox: ReleaseSandbox, change: str) -> None:
    scenario = replace(sandbox, environment={**sandbox.environment, "RUNS_FILTER": change})
    result = scenario.run(START)
    assert result.returncode != 0, result.stdout
    assert not any("--method POST" in request for request in sandbox.requests())


@pytest.mark.parametrize("version", ["0.1.3", "null"])
def test_start_release_requires_matching_manifest(sandbox: ReleaseSandbox, version: str) -> None:
    _ = (sandbox.root / ".release-please-manifest.json").write_text(f'{{".": "{version}"}}')
    assert sandbox.run(START).returncode != 0
    assert not any("--method POST" in request for request in sandbox.requests())


def test_start_release_rejects_a_commit_outside_main(sandbox: ReleaseSandbox) -> None:
    unrelated = git(sandbox.root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit-tree", "HEAD^{tree}", "-m", "unrelated main history").strip()
    _ = git(sandbox.root, "update-ref", "refs/remotes/origin/main", unrelated)
    assert sandbox.run(START).returncode != 0
    assert not any("--method POST" in request for request in sandbox.requests())


def test_start_release_keeps_the_reviewed_commit_when_main_advances(sandbox: ReleaseSandbox) -> None:
    later = git(sandbox.root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "later main commit").strip()
    _ = git(sandbox.root, "update-ref", "refs/remotes/origin/main", later)
    result = sandbox.run(START)
    assert result.returncode == 0, result.stderr
    assert sandbox.requests()[-1].endswith(f"-f sha={sandbox.environment['EXPECTED_SHA']}")


def test_start_release_does_not_reuse_a_tag(sandbox: ReleaseSandbox) -> None:
    _ = git(sandbox.root, "tag", "v0.2.0")
    assert sandbox.run(START).returncode != 0
    assert not any("--method POST" in request for request in sandbox.requests())


@pytest.mark.parametrize("endpoint", ["/pulls/41", "/actions/workflows/ci.yml/runs", "/git/refs"])
def test_start_release_propagates_api_failures(sandbox: ReleaseSandbox, endpoint: str) -> None:
    scenario = replace(sandbox, environment={**sandbox.environment, "FAIL_REQUEST": endpoint})
    assert scenario.run(START).returncode != 0


@pytest.mark.parametrize(("actor", "triggering_actor"), [
    ("", "Korridzy"), ("99", "Korridzy"), ("7", ""), ("7", "other"),
])
def test_start_release_requires_the_owner_before_any_lookup(
    sandbox: ReleaseSandbox, actor: str, triggering_actor: str,
) -> None:
    scenario = replace(sandbox, environment={
        **sandbox.environment, "GITHUB_ACTOR_ID": actor, "GITHUB_TRIGGERING_ACTOR": triggering_actor})
    assert scenario.run(START).returncode != 0
    assert sandbox.requests() == []


def test_release_labels_require_successful_publication() -> None:
    job = mapping(mapping(workflow("release")["jobs"])["release-labels"])
    assert job["needs"] == ["github-release"]
    assert "if" not in job and "continue-on-error" not in job
    assert job["permissions"] == {"contents": "read", "pull-requests": "read", "issues": "write"}
    assert mapping(steps("release-labels", "release")["labels"]["env"]) == {"GH_TOKEN": "${{ github.token }}"}


def test_release_labels_clear_pending_only_after_tagged_is_added(sandbox: ReleaseSandbox) -> None:
    result = sandbox.run(LABELS)
    assert result.returncode == 0, result.stderr
    assert sandbox.requests()[1:] == [
        "api --method POST repos/Korridzy/sttx/issues/41/labels -f labels[]=autorelease: tagged",
        "api --method DELETE repos/Korridzy/sttx/issues/41/labels/autorelease%3A%20pending"]


@pytest.mark.parametrize("change", [
    ".[1] = []", '.[1][0].merge_commit_sha = "wrong"', '.[1][0].base.ref = "feature"',
    '.[1][0].head.ref = "feature"', '.[1][0].head.repo.full_name = "other/repo"',
    ".[1][0].merged_at = null", '.[1][0].user.type = "User"',
    '.[1][0].user.login = "Korridzy"', '.[1][0].user.login = "other[bot]"', ".[1][0].user.login = null",
])
def test_release_labels_leave_unrelated_prs_untouched(sandbox: ReleaseSandbox, change: str) -> None:
    scenario = replace(sandbox, environment={**sandbox.environment, "PULLS_FILTER": change})
    assert scenario.run(LABELS).returncode == 0
    assert len(sandbox.requests()) == 1


def test_release_labels_reject_ambiguous_matches(sandbox: ReleaseSandbox) -> None:
    scenario = replace(sandbox, environment={**sandbox.environment, "PULLS_FILTER": ".[1] += .[1]"})
    assert scenario.run(LABELS).returncode != 0
    assert len(sandbox.requests()) == 1


def test_release_labels_can_be_retried_after_cleanup(sandbox: ReleaseSandbox) -> None:
    scenario = replace(sandbox, environment={
        **sandbox.environment, "PULLS_FILTER": '.[1][0].labels = [{"name":"autorelease: tagged"}]'})
    assert scenario.run(LABELS).returncode == 0
    assert not any("--method DELETE" in request for request in sandbox.requests())


def test_release_labels_keep_pending_when_tagged_update_fails(sandbox: ReleaseSandbox) -> None:
    scenario = replace(sandbox, environment={**sandbox.environment, "FAIL_REQUEST": "--method POST"})
    assert scenario.run(LABELS).returncode != 0
    assert not any("--method DELETE" in request for request in sandbox.requests())
