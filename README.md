# sttx

`sttx` is a standalone Linux CPU transcription command. It normalizes one
audio/video file with `ffmpeg`, runs NVIDIA Parakeet TDT 0.6B v3 through the
`sherpa-onnx` Python bindings, and writes timestamped JSON and plain text
transcripts.

## Installation

`sttx` officially supports Linux and requires Python 3.11, 3.12, or 3.13. Its
native dependencies also publish macOS and Windows packages, but `sttx` uses
Unix process-group APIs and its process lifecycle is currently tested only on
Linux.

### Install ffmpeg

Install `ffmpeg` separately and make sure it is available on `PATH`. The
application invokes it for every input, including WAV files that already have
the target format.

### Install sttx from PyPI

Install the published [`sttx` package from PyPI](https://pypi.org/project/sttx/):

```bash
python -m pip install sttx
sttx --version
```

This installs the `sttx` console command and its Python runtime dependencies.

## Quickstart

```bash
sttx recording.mp4
```

The default output directory is `./transcriptions`. For `recording.mp4`, a
successful run prints these two paths to stdout and writes both files:

```text
transcriptions/recording.json
transcriptions/recording.txt
```

Existing output files are replaced atomically.

## Command-line interface

The complete public surface is:

```text
sttx [-h] [-o STEM] [-d DIR] [--model-dir DIR] [-ts] [-v] [--debug] [--log-format {text,json}] [--version] media
```

- `media` — required positional path to one local audio/video file.
- `-h`, `--help` — show usage and exit.
- `-o STEM`, `--output-name STEM` — output filename stem. The program adds
  `.json` and `.txt`; a path or an existing suffix is rejected.
- `-d DIR`, `--outdir DIR` — output directory. Relative paths are
  resolved from the current directory; the default is `./transcriptions`.
- `--model-dir DIR` — use a local, offline model bundle instead of
  acquiring model assets.
- `-ts`, `--timestamp` — write each TXT line as
  `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`.
  This affects TXT contents only; JSON is unchanged.
- `-v`, `--verbose` — write elapsed-time pipeline stage messages and live
  transcription progress to stderr. Text logs include a 20-character ASCII
  bar, processed audio position, real-time factor, estimated time remaining,
  and a final duration/real-time-factor summary. In an interactive terminal,
  an elapsed-time prefix and progress bar redraw in place ten times per second;
  redirected stderr receives readable line-oriented progress snapshots.
- `--debug` — enable pipeline progress plus diagnostic details on stderr:
  ffmpeg arguments, the normalized WAV path, model source and asset paths/sizes,
  stage timings, live ASR scan/VAD/decode activity, reported language, word
  counts, and an aggregate ASR summary. It includes a traceback for unexpected
  failures and never prints transcript content.
- `--log-format {text,json}` — format enabled stderr diagnostics as human text
  (the default) or JSON Lines. JSON records have stable `event` and
  `elapsed_seconds` fields plus event-specific data such as `stage`,
  `audio_seconds`, `percent`, and `eta_seconds`; use it with `-v` or `--debug`
  for machine-consumable pipeline telemetry.
- `--version` — print `sttx 0.1.3` and exit. <!-- x-release-please-version -->

There is no language option: the model reports a language when available and
the JSON writer falls back to `"auto"`.

## Streams and exit codes

On a successful transcription, stdout contains exactly two newline-separated
paths, JSON first and TXT second. The application writes diagnostics to
stderr: argument usage/errors, environment or runtime errors, the no-speech
warning, optional verbose stage and live-progress messages, and debug
diagnostics. `--help` and `--version` are the usual argparse exceptions: their
informational text is printed to stdout and they exit successfully.

With `--log-format json`, enabled diagnostic records are one JSON object per
stderr line. Successful invocations still print only the two output paths to
stdout. Argument usage, `--help`, `--version`, and signal cancellation retain
their normal argparse or cancellation text behavior.

The process exits with:

- `0` — transcription completed, including a silence/no-speech input.
- `1` — input or environment failure (for example an unreadable file,
  unavailable `ffmpeg`, missing model asset, or output filesystem failure).
- `2` — argument/usage failure or transcription/decode failure.
- `130` — clean cancellation after SIGINT.
- `143` — clean cancellation after SIGTERM.

Errors are emitted on stderr with an `error:` prefix. A cancellation also
prints a short cancellation diagnostic on stderr. Temporary normalized audio is
removed during success, failure, and signal cleanup.

## Model acquisition and cache

Without `--model-dir`, sttx resolves
`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` at the pinned revision
`2bda32ec70b097a55adaa07d9a7173915b43cc78`. Both local-only lookup and network
acquisition request that exact revision, not the repository's current head.
Missing Parakeet assets are downloaded into the Hugging Face cache.

Acquisition is local-first: a complete local snapshot of the pinned revision
is reused without a refresh request. Network acquisition is attempted only
when that snapshot is missing or incomplete. Runtime checks require readable,
nonempty Parakeet assets; the large model files are not rehashed on every run.
Qualification separately checks their declared SHA-256 identities.

Silero VAD is stored at `~/.cache/sttx/silero_vad.onnx`. Its SHA-256 is checked
before reuse. A missing, unreadable, empty, or hash-mismatched cache entry
triggers reacquisition. Downloaded bytes are staged and verified against the
[backend contract](src/sttx/backend_contract.py) before atomic replacement.
A bad download fails rather than replacing an existing final file; its staging
file is cleaned up. A valid warm Silero cache needs no download.

For an offline run, `--model-dir` must point to a flat directory containing the
complete five-file bundle below. These exact filenames are required and each
file must be readable and non-empty; extra files are not consumed:

```text
encoder.int8.onnx
decoder.int8.onnx
joiner.int8.onnx
tokens.txt
silero_vad.onnx
```

Explicit bundles remain zero-network and use the same filename, readability,
and nonempty checks as before. They are not subjected to the default model's
revision or checksum policy and do not participate in automatic cache repair.
Arbitrary offline bundles and explicit dependency overrides are outside the
qualification guarantees.

## Output files

The JSON file is invariant whether or not you pass `-ts` or `--timestamp`. It is UTF-8, uses
two-space indentation, and keeps non-ASCII text unescaped. Its `text` field is
the segment text joined with single spaces. Every object in `segments` has
numeric `start` and `end` fields, rounded to two decimal places. `duration` is
also rounded to two decimal places, and segment IDs start at zero.

For example, one short transcript may be:

```json
{
  "task": "transcribe",
  "language": "auto",
  "duration": 3.2,
  "text": "Hello world.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Hello world."
    }
  ]
}
```

Without `-ts` or `--timestamp`, the matching TXT file contains one normalized non-empty
segment per line, with no trailing newline. JSON `text` remains the same segments joined
with single spaces:

```text
Hello world.
```

With `-ts` or `--timestamp`, TXT contains one normalized non-empty segment per line. Each line
uses raw segment timing rounded to milliseconds; hours are unbounded and use at
least two digits. The file has no trailing newline:

```text
[00:00:00.000 --> 00:00:03.200] Hello world.
```

`-ts` and `--timestamp` affect TXT contents only. They do not change JSON, including the
top-level `text` field or the numeric `start` and `end` fields in `segments`.

Silence/no speech is a successful result with `text: ""`, an empty `segments`
array, and `warning: no speech detected` on stderr. In either TXT mode, it
writes an empty TXT file.

## Development

From a source checkout, Poetry creates the project's virtual environment in
`.venv/`:

```bash
poetry install
poetry env info --path
poetry check
poetry run basedpyright
poetry run pytest -q -m "not integration"
poetry run python -m compileall -q src tests
poetry run sttx --help
poetry run sttx --version
poetry build
```

The offline suite uses fakes for native inference and does not download models.
An unfiltered `poetry run pytest` also runs the three real integration gates;
these require native packages, models/network, `ffmpeg`, `strace`, and working
Linux namespace support. Keep their evidence and build outputs in scratch
directories, not tracked fixtures. `poetry.lock` remains ignored local state.
The [repository contract](tests/test_repo_contract.py) requires the approved
qualification sources, tests, scripts, baseline JSON, and this gap history;
the [packaging contract](tests/test_packaging.py) still requires an exact pure
wheel inventory and exact dependency metadata. Models, media, run evidence,
wheels, caches, and lockfiles are not repository deliverables.

### Backend qualification identity

Normal fresh installations resolve the four exact direct runtime pins in
`pyproject.toml`: `sherpa-onnx==1.13.8`, `sherpa-onnx-bin==1.13.8`,
`huggingface-hub==2.1.1`, and `numpy==2.4.6`. This is not a lock of every
transitive dependency or a promise of identical numerics on every CPU.

The qualification fingerprint includes sorted direct dependencies, the backend
contract payload (settings, asset and independently pinned fixture identities,
timing constants, probe version, and approved-run anchor), and the complete
bytes of exactly these 26 required sources:

```text
scripts/compute_backend_fingerprint.py
scripts/backend_change_detector.py
tests/conftest.py
src/sttx/asr.py
src/sttx/cli.py
src/sttx/model.py
src/sttx/audio.py
src/sttx/asr_events.py
tests/integration/timing_lattice.py
tests/integration/test_timing_qualification.py
tests/integration/qualification_schema.py
tests/integration/qualification_validation.py
tests/integration/qualification_evaluation.py
tests/integration/qualification_recording.py
tests/integration/qualification_probes.py
tests/integration/test_binding_contract.py
tests/integration/test_real_pipeline.py
tests/integration/real_pipeline_artifacts.py
tests/integration/real_pipeline_checks.py
tests/integration/real_pipeline_evidence_contract.py
tests/integration/real_pipeline_media.py
tests/integration/real_pipeline_observability.py
tests/integration/real_pipeline_runner.py
tests/integration/real_pipeline_signal_process.py
tests/integration/real_pipeline_signals.py
tests/integration/real_pipeline_trace.py
```

Any bound-byte change, including comments, invalidates the old fingerprint.
All 26 sources must be present for qualification and release. New executable
qualification logic must join the binding before use, not escape into an
unhashed helper. Ownership is deliberate:

| Owner | Responsibility |
| --- | --- |
| `scripts/compute_backend_fingerprint.py` | Read-only checker CLI and promotion policy |
| `scripts/backend_change_detector.py` | Base/head AST and diff decisions, without executing base code |
| `tests/integration/test_timing_qualification.py` | Native gate lifecycle and failure envelopes |
| `qualification_schema.py`, `qualification_validation.py`, `qualification_evaluation.py` | Typed serialization, parsing/validation, and pure comparison respectively |
| `qualification_recording.py`, `qualification_probes.py` | Passive production recording and real probe collection respectively |
| `src/sttx/model.py`, `src/sttx/audio.py`, `src/sttx/asr_events.py` | Asset acquisition and checksums, ffmpeg normalization of the observed samples, and the activity-event union the probes consume |
| `tests/integration/test_binding_contract.py`, `test_real_pipeline.py`, `real_pipeline_*.py` | Native binding and real pipeline gates, their fixtures, checks, and evidence |
| `tests/conftest.py` | Pytest options and atomic evidence writer |
| `tests/backend_qualification_helpers.py` | Offline synthetic fixtures only; never a checker/gate runtime import |

Bare `qualification_*.py` names refer to files under `tests/integration/`.
The signed observation payload has three separate collections: raw lattice
`runs`, VAD/chunk-level `production_runs`, and final `production_transcripts`.
Actual production builders and `transcribe()` supply nonempty speech and
multichunk coverage. This finite fixture and its timing tolerances are an
acceptance envelope, not proof against every numerical or transcription change.
See [KNOWN-GAPS.md](KNOWN-GAPS.md) for historical context and residual limits.

Read-only checks from the repository root:

```bash
poetry run python scripts/compute_backend_fingerprint.py --help
poetry run python scripts/compute_backend_fingerprint.py
poetry run python scripts/compute_backend_fingerprint.py --check tests/integration/qualified_backend.json
```

Print mode emits an identity, not a qualification verdict. `--check` validates
the baseline, signature/anchor, provenance, and current source identity; it does
not run inference or certify a clean Git worktree. Neither mode writes a record.

### Qualifying a backend update locally

Complete and review all contract, dependency, and bound-source changes before
collecting candidates. A changed bound probe source requires an increasing
`PROBE_VERSION` and a changed qualified record. Preserve the previous baseline
and anchor together before changing either. Keep C1/C2/C3 in distinct files in
one persistent local package environment; CI candidates are diagnostic only and
must never be promoted directly.

1. Allocate a fresh evidence directory, for example
   `QUALIFICATION_DIR=$(mktemp -d /tmp/sttx-qualification.XXXXXX)`. Measure C1
   against the previous baseline and anchor with the command below. Keep C1
   unchanged, even if the final aggregate assertion is nonzero. Inspect the
   complete candidate, not merely that exit status: every candidate must match
   the current contract. For routine updates C1 must be comparable with empty
   integrity diagnostics. Only reviewed stale fingerprint, probe-version,
   model-repo/revision/assets, dependencies, or quantum identity diagnostics are
   eligible; verify each delta against the intended change.

   ```bash
   poetry run pytest tests/integration/test_timing_qualification.py -m integration -q --qualification-output="$QUALIFICATION_DIR/C1.json"
   ```

2. Review all behavioral differences. Routine promotion requires empty behavior
   diagnostics; accepting a known difference is a deliberate maintainer decision
   using `--accept-behavior-drift`, with the reason recorded in review. Never use
   it to bypass invalid observations, missing coverage, fixture/platform mismatch,
   integrity failures, or unknown diagnostic codes.

3. Rotate `QUALIFIED_RUNS_SHA256` to C1's signed-payload signature and rerun the
   same gate into `"$QUALIFICATION_DIR/C2.json"`. C2 must have the current
   fingerprint and anchor and exactly the same canonical three-collection
   payload and signature as C1. Only the permitted anchor-transition diagnostic
   may be added to the reviewed identity diagnostics. **Stop on instability.**
   Run the read-only promotion check before writing any baseline:

   ```bash
   poetry run pytest tests/integration/test_timing_qualification.py -m integration -q --qualification-output="$QUALIFICATION_DIR/C2.json"
   poetry run python scripts/compute_backend_fingerprint.py --check-promotion "$QUALIFICATION_DIR/C1.json" "$QUALIFICATION_DIR/C2.json"
   ```

4. After checker success, promote C2's observations and identity into
   `tests/integration/qualified_backend.json` with `record_kind=qualified_backend`.
   Replace its top-level `baseline_comparison` with C1's comparison and add
   `promotion={previous_fingerprint,previous_anchor,comparison_to_previous,decision}`.
   The provenance comparison must equal the top-level comparison. Use `routine`
   or `accepted_behavior` as appropriate; preserve the actual previous identity.
   The checker does not perform this edit for you.

5. Run `--check tests/integration/qualified_backend.json`, then the native gate
   again into `"$QUALIFICATION_DIR/C3.json"`. C3 must pass with empty diagnostics.
   Land the reviewed baseline and anchor together, never an intermediate state.
   On any failed rotation, restore the saved baseline/anchor pair and retain
   candidate evidence for diagnosis. Any subsequent bound-source edit requires
   reviewed regeneration and the full sequence again.

   ```bash
   poetry run python scripts/compute_backend_fingerprint.py --check tests/integration/qualified_backend.json
   poetry run pytest tests/integration/test_timing_qualification.py -m integration -q --qualification-output="$QUALIFICATION_DIR/C3.json"
   ```

Bootstrap is only for a genuinely absent baseline: C1/C2 may report only
`baseline.missing`, with otherwise valid observations. Use `--bootstrap` on the
pair check, then `decision=bootstrap` with null previous fingerprint/anchor.
It is incompatible with `--accept-behavior-drift` and refuses an existing
baseline; deleting a baseline is not an upgrade bypass. Failure envelopes are
never promotable. Unsafe output aliases or filesystem failures can prevent
artifact creation and still fail the gate.

### CI, dependency updates, and releases

The [CI workflow](.github/workflows/ci.yml) retains the Python 3.11-3.13 offline
matrix. On unrelated PRs, the backend job explicitly succeeds without model
downloads, native gates, or artifact uploads. The detector requires qualification
when the fingerprint mismatches the committed baseline or the record changed;
metadata-only `pyproject.toml` edits are not automatically expensive. Missing
base history follows a strict first-introduction policy, not silent fast-pass.

The slow path runs qualification, binding, and the full pipeline separately,
retains each outcome, uploads every required evidence file individually, and
fails the aggregate on any failed/skipped gate, missing upload, or stale
baseline. The stable `checks` job requires both the matrix and backend job.
On fresh jobs, qualification acquires an ambient bundle, binding reuses it, and
the full pipeline acquires an independently isolated cold bundle with real
signal/cache checks. No Actions model cache persists across jobs or runs.

[Dependabot](.github/dependabot.yml) proposes weekly grouped speech-backend
updates. There is no auto-merge, automatic baseline commit, or auto-promotion.

An authorized `v*` tag starts the [release workflow](.github/workflows/release.yml).
The tag must match the package version. Build runs once; qualification downloads
that exact artifact ID and installs its wheel into a fresh non-editable venv,
proves import origin, then runs all three native gates before publication.
Publish and GitHub release download that same artifact through the dependency
chain `build -> qualify -> publish -> github-release`; there is no rebuild.
OIDC `id-token: write` is confined to publish under the `pypi` environment, and
`contents: write` to the GitHub release job with `GH_TOKEN` bound to the workflow
token. Failed qualification blocks publication. These are implemented workflow
contracts, not evidence of hosted execution. No tag or publication is part of
local qualification.

### Release Please maintainer setup

Release Please only opens or updates one release PR against `main`, containing
the proposed version and changelog. It does not create tags, GitHub releases,
or PyPI publications. A maintainer separately starts the
[Start release workflow](.github/workflows/start-release.yml) to tag an approved,
merged release PR; the existing release pipeline handles publication. The reviewed
baseline
`cb158c19149795d5a98cb3e83bf2c0f8c9a3024a` is both `bootstrap-sha` and the
temporary `last-release-sha`, with manifest version `0.1.3`. Together they
prevent historical changelog regeneration: the existing `v0.1.3` tag is outside
main's ancestry, so bootstrap alone is insufficient. Existing `v0.1.3` is
immutable; do not move it to repair discovery. After the first correctly tagged
main release is discoverable, remove the temporary `last-release-sha` override
in a reviewed change so future discovery uses the actual release boundary.

Before enabling the workflow:

1. Sign in as the repository owner, **Korridzy**, and open account
   **Settings -> Developer settings -> Personal access tokens -> Fine-grained
   tokens -> Generate new token**. Choose an expiration, set **Resource owner**
   to **Korridzy**, and select **Only select repositories -> sttx**.
   Grant repository **Contents: read and write**, **Pull requests: read and
   write**, and **Issues: read and write**; Metadata read access is automatic.
   No GitHub App registration, client ID, or private key is needed.
2. Store the PAT in repository **Settings -> Secrets and variables -> Actions
   -> New repository secret**, named `RELEASE_PLEASE_TOKEN`. Paste the token
   value, not a filename. Alternatively, use the interactive CLI prompt:

   ```bash
   gh secret set RELEASE_PLEASE_TOKEN --repo Korridzy/sttx
   gh secret list --repo Korridzy/sttx
   ```

   Replace this secret when rotating the token, before its expiration.
   Both Release Please and Start release check the authenticated account's
   numeric ID against the repository owner's ID before writing. Their release
   PRs are authored by **Korridzy**, not an App bot; the launcher and label
   cleanup validate that same owner ID as well as the release branch.
3. Preserve normal branch protections and required checks. The PAT permits
   generated PRs to trigger normal CI and its tag pushes to trigger the separate
   Release workflow. `GITHUB_TOKEN`-created PRs can require workflow approval,
   and its tag pushes do not trigger that push-based release workflow.
   Read-only launcher validation still uses `GITHUB_TOKEN`, including Actions
   read access for CI results. Post-publication label cleanup uses its narrowly
   scoped workflow token; the PAT is not passed to the publication jobs.

An authorized, controlled hosted run must verify PAT authentication, one release
PR, synchronized versions, PR-triggered CI, and no automatic tag/release/publish
from the Release Please workflow. For a nonpublishing smoke test, merge the
implementation with a releasable `feat:` squash title, check the proposed release
PR and its CI, then leave that PR open without running Start release. A `ci:` or
`chore:` squash title alone intentionally produces no release PR.

Verify the separate manual launcher and its downstream pipeline when publishing
an approved release. Parsed configuration and local tests, including controlled
GitHub fixtures, are local evidence only; they do not prove PAT availability,
permissions, or hosted execution.

### Reviewed release PR to maintainer publication

Use Conventional Commit PR titles and preserve them as squash-merge titles:
`fix: ...` proposes a patch, `feat: ...` a minor, and `docs: ...` or `chore: ...`
alone do not propose a release unless accompanied by a releaseable change.
Mark breaking changes with `!` or a `BREAKING CHANGE:` footer in the resulting
commit. The explicit pre-1.0 policy makes breaking changes minor, not major:
from the existing `0.1.3`, a fix becomes `0.1.4`, while a feature or breaking
change becomes `0.2.0`. After 1.0, breaking changes bump the major version.

1. Merge reviewed changes to `main`. Release Please creates or updates its
   single release PR. Review its changelog and synchronized versions in
   `pyproject.toml`, `src/sttx/__init__.py`, packaging expectations, README CLI
   example, and manifest. The root Python strategy updates project metadata;
   `x-release-please-version` annotations keep generic extra-file values
   synchronized without weakening exact wheel inventory or metadata tests.
   Historical references to existing releases are not updater targets.
2. Validate the release PR locally with the development checks above, including
   `poetry run pytest -q tests/test_packaging.py`. Do not assume a backend
   first-introduction result: the current detector decides whether qualification
   is needed. Backend-affecting changes still require the full local
   qualification cycle above; the hosted slow path runs on Python 3.13 and
   requires working `unshare`. Cross-interpreter identity or behavior drift must
   be resolved, not bypassed.
3. Require the unchanged Python 3.11-3.13 `test` matrix, `backend-compat`, and
   stable `checks` job on the release PR. Review and merge to `main` only when
   green under normal protections. There is no automatic merge or protection
   bypass.
4. Wait for the merged commit's push-to-`main` CI run to succeed. Open
   **Actions -> Start release -> Run workflow**, select **main**, and enter the
   merged Release Please PR number. The workflow must first be merged into the
   default branch for the button to appear. No local tag commands are needed.

   The launcher accepts only a merged, repository-local release PR authored by
   the repository owner, targeting `main` with `autorelease: pending`. It checks
   out that exact merge commit,
   requires it to remain in main's history, verifies matching package and manifest
   versions, rejects an existing tag, and requires successful main CI for that
   exact commit. Only then does it authenticate the owner's PAT and create
   `v<version>`.
   Concurrent launches are serialized; tag creation never replaces an existing
   ref. Its run summary links to the separate Release workflow.

   If CI is still running or a validation fails, resolve that condition before
   running Start release again. Never retag `v0.1.3`, reuse a published version,
   or publish from a feature branch. PyPI does not permit version reuse;
   a replacement publication needs a new reviewed version and new tag.
5. Monitor the existing `build -> qualify -> publish -> github-release` workflow.
   It qualifies and publishes the same built distributions, with no rebuild.
   A failed qualification blocks publication; environment approvals and normal
   release protections still apply.
6. After successful PyPI publication and GitHub release creation, the separate
   `release-labels` job finds the merged release PR for the tagged commit, adds
   `autorelease: tagged`, then removes `autorelease: pending`. Other labels are
   preserved. Release Please otherwise blocks subsequent release PRs while a
   merged release PR is pending; its own release-creation phase remains disabled.

   If only label cleanup fails, rerun that failed job rather than retagging or
   republishing. The cleanup can be retried without repeating publication.
   Failed publication leaves pending in place. Legacy tags without an associated
   Release Please PR need no label update. Label cleanup alone does not trigger
   Release Please; run it manually afterward, or let the next push to `main`
   trigger it.

## Attribution and licensing

`sttx` source code is licensed under the [GNU General Public License v3.0 only](LICENSE).
Third-party software and model assets remain subject to their own terms; see
[NOTICE.md](NOTICE.md) for attribution and primary upstream license links.
