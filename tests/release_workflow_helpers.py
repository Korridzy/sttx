from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .test_backend_workflows import YamlValue
from .test_repo_contract import git

FAKE_GH: Final = r'''
gh() {
  printf '%s\n' "$*" >> "$API_CALLS"
  if [[ -n "$FAIL_REQUEST" && "$*" == *"$FAIL_REQUEST"* ]]; then return 1; fi
  case "$*" in
    "api repos/Korridzy/sttx/pulls/41")
      printf '%s\n' "$PR_JSON" | jq "$PR_FILTER" ;;
    "api --method GET repos/Korridzy/sttx/actions/workflows/ci.yml/runs -f head_sha=$EXPECTED_SHA -f event=push -f branch=main -f per_page=1")
      printf '%s\n' "$RUNS_JSON" | jq "$RUNS_FILTER" ;;
    "api --method POST repos/Korridzy/sttx/git/refs -f ref=refs/tags/v0.2.0 -f sha=$EXPECTED_SHA")
      printf '{}\n' ;;
    "api repos/Korridzy/sttx/commits/$EXPECTED_SHA/pulls --paginate --slurp")
      printf '%s\n' "$PULLS_JSON" | jq "$PULLS_FILTER" ;;
    "api --method POST repos/Korridzy/sttx/issues/41/labels -f labels[]=autorelease: tagged")
      printf '[]\n' ;;
    "api --method DELETE repos/Korridzy/sttx/issues/41/labels/autorelease%3A%20pending")
      printf '[]\n' ;;
    *) printf 'Unexpected API request: %s\n' "$*" >&2; return 97 ;;
  esac
}
'''


@dataclass(frozen=True, slots=True)
class ReleaseSandbox:
    root: Path
    environment: Mapping[str, str]

    def run(self, scripts: tuple[YamlValue, ...]) -> subprocess.CompletedProcess[str]:
        """Execute shipped shell steps with real git/Poetry/jq and a closed API."""
        environment = {**os.environ, **self.environment}
        result: subprocess.CompletedProcess[str] = subprocess.CompletedProcess([], 0, "", "")
        for script in scripts:
            assert isinstance(script, str)
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", FAKE_GH + script],
                cwd=self.root, env=environment, capture_output=True, text=True, timeout=30,
            )
            if result.returncode:
                return result
            outputs = dict(line.split("=", 1) for line in (self.root / "outputs").read_text().splitlines())
            environment.update(RELEASE_SHA=outputs.get("sha", ""), RELEASE_TAG=outputs.get("tag", ""))
        return result

    def requests(self) -> list[str]:
        return (self.root / "requests").read_text().splitlines()


def release_sandbox(root: Path) -> ReleaseSandbox:
    _ = (root / "pyproject.toml").write_text('[project]\nname = "sttx"\nversion = "0.2.0"\n')
    _ = (root / ".release-please-manifest.json").write_text('{".": "0.2.0"}')
    _ = git(root, "init", "--quiet", "-b", "main")
    _ = git(root, "add", "pyproject.toml", ".release-please-manifest.json")
    _ = git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
            "commit", "--quiet", "-m", "release fixture")
    sha = git(root, "rev-parse", "HEAD").strip()
    _ = git(root, "update-ref", "refs/remotes/origin/main", sha)
    for name in ("outputs", "requests", "summary"):
        (root / name).touch()
    pr = {
        "number": 41, "merged": True, "merged_at": "2026-10-22T00:00:00Z", "merge_commit_sha": sha,
        "base": {"ref": "main", "repo": {"full_name": "Korridzy/sttx"}},
        "head": {"ref": "release-please--branches--main", "repo": {"full_name": "Korridzy/sttx"}},
        "user": {"type": "Bot"}, "labels": [{"name": "autorelease: pending"}, {"name": "keep"}],
    }
    runs = {"workflow_runs": [
        {"head_sha": sha, "head_branch": "main", "event": "push", "status": "completed", "conclusion": "success"}
    ]}
    return ReleaseSandbox(root, {
        "GITHUB_REPOSITORY": "Korridzy/sttx", "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": sha, "GITHUB_OUTPUT": str(root / "outputs"),
        "GITHUB_STEP_SUMMARY": str(root / "summary"), "GH_TOKEN": "fixture",
        "RELEASE_PR": "41", "RELEASE_SHA": "", "RELEASE_TAG": "", "EXPECTED_SHA": sha,
        "PR_JSON": json.dumps(pr), "RUNS_JSON": json.dumps(runs), "PULLS_JSON": json.dumps([[], [pr]]),
        "PR_FILTER": ".", "RUNS_FILTER": ".", "PULLS_FILTER": ".", "FAIL_REQUEST": "",
        "API_CALLS": str(root / "requests"),
    })
