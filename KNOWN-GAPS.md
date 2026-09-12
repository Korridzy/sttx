# Known Gaps

Historical failure context, implemented safeguards, and remaining evidence limits.

## 1. Detecting Speech Backend Changes

**Raised:** 2026-08-31, while fixing `token end lies outside its chunk`
(`fix(asr): clamp frame-quantized token timings to their chunk`).

**Status:** qualification mechanisms implemented; finite-fixture limitations and
full release-mirror/hosted execution verification remain open.

### Historical Failure

The failure came from assuming that sherpa-onnx's `timestamp + duration` always
stays within the decoded chunk. Frame-quantized timing could violate that
assumption, causing the `token end lies outside its chunk` error. A functional
test with hand-written fake results cannot establish the real backend's
numerical timestamp and duration semantics. An import check can detect a removed
API, but not this kind of behavioral change.

At the time, `sherpa-onnx`, `sherpa-onnx-bin`, `huggingface-hub`, and `numpy` had
no version constraints, and the ignored Poetry lockfile was forbidden from the
tracked repository. A fresh resolve could therefore change backend behavior
without a repository diff. CI excluded the two real integration tests. A
deliberate model swap was reviewable but did not force measurement of timing
assumptions; a silent upstream dependency update had no review event at all.
Either could leave CI green while the tool crashed or produced wrong timings.

The original note called the model "pinned"; that wording described its assumed
identity, not an enforced runtime snapshot revision. The pre-change downloader
was floating. Explicit revision enforcement is one of the mechanisms below.
The model's documented subsampling factor of 8 over a 10 ms feature hop motivated
the 0.08 s encoder frame assumption; the current 1.0 s overhang ceiling is a
separate contract limit, not a universal consequence of that factor.

The historical trade-off was real: cold acquisition deliberately uses an
isolated cache, so persisting models to make every PR cheap would defeat that
coverage. The original note reported roughly 640 MB per cold model acquisition
and 130-145 s for both integration tests locally on a warm system. Those are
dated 2026-08-31 observations, not current runner time or bandwidth promises.

**Provenance:** this file recreates the former untracked root note, which the
user requested be deleted before implementation began. The failure description
above is a faithful summary of its 118-line read captured immediately before
authorized deletion in OpenCode session `ses_f8d2409eaffe0a5p34ZB0fDtxT`, read
tool call `call_0j8vMYQiWsDiAexJLmIv3ryy` on 2026-09-05. Historical claims are
not presented as the current state.

### Implemented Mechanisms

| Original goal | Current mechanism |
| --- | --- |
| Signal backend timing changes before adoption | [Exact runtime pins](pyproject.toml), [real qualification](tests/integration/test_timing_qualification.py), and [blocking CI/release dependencies](.github/workflows/release.yml) make upgrades explicit and test observed behavior. |
| Make a model swap require conscious timing re-evaluation | The [shipped contract](src/sttx/backend_contract.py) names model revision/hashes, shared production settings, independent fixture identity, timing constants, and anchor. The [26-source complete-byte fingerprint](scripts/compute_backend_fingerprint.py) invalidates stale qualification; reviewed local regeneration is mandatory. |
| Avoid recurring native-gate cost on ordinary PRs | The [AST/diff detector](scripts/backend_change_detector.py) and [CI fast path](.github/workflows/ci.yml) skip native installs/gates/uploads when the backend fingerprint and record are unchanged. |
| Preserve cold acquisition | The [real pipeline](tests/integration/test_real_pipeline.py) keeps isolated cold/warm and signal probes. Qualification/binding may share only the job-local ambient cache; workflows do not persist model caches. |

[Model resolution](src/sttx/model.py) requests the exact default revision on both
local-first and network paths. Silero cache reuse checks SHA-256, repairs bad
entries by reacquisition, and verifies staged bytes before replacement. Explicit
five-file offline bundles retain their zero-network/readable/nonempty contract;
they are not forced to match the default model hashes.

Qualification measures raw lattice runs, actual production VAD/chunk runs, and
final transcripts as three signed collections. It requires nonempty speech and
an actual long VAD segment exercising the 480000-sample split. The five approved
helpers own [schema](tests/integration/qualification_schema.py),
[validation](tests/integration/qualification_validation.py),
[evaluation](tests/integration/qualification_evaluation.py),
[recording](tests/integration/qualification_recording.py), and
[probe collection](tests/integration/qualification_probes.py). No offline test
fixture is a checker or gate runtime dependency.

The [local promotion protocol](README.md#qualifying-a-backend-update-locally)
preserves C1's old-baseline comparison, rotates the anchor, checks C2's exact
signed-payload stability, records previous fingerprint/anchor and the maintainer
decision, then requires C3 against the final pair. Instability stops promotion.
Invalid candidates, failure envelopes, unknown diagnostics, and fixture/platform
mismatches cannot be accepted by a switch. Behavior drift requires explicit
acceptance, not bootstrap. CI artifacts are diagnostic only, never automatically
promoted. Failed rotations restore the old baseline and anchor together.

### Residual Limits and Verification Status

- This is a finite English fixture plus silence-prefix and synthetic long-probe
  envelope, not a multilingual corpus or exhaustive numerical equivalence proof.
  An observed timestamp gcd can alias a multiple of the true lattice. The
  [comparator](tests/integration/timing_lattice.py) quantizes to 10 microseconds,
  permits bounded drift up to 80000 microseconds, and limits material changes in
  the 40000-80000 band to `max(1, (2*n)//100)` per compared timing array. Changes
  outside these observations or within allowed tolerances may go undetected.
- Exact direct dependency pins do not lock transitive packages, OS libraries,
  ffmpeg, hardware, or every numerical kernel. Platform identity records system
  and machine, not all CPU/runtime characteristics. Qualification of a local
  environment is not proof for all supported Python versions and hosts.
- The default Parakeet runtime validates files without rehashing large assets
  every invocation. Qualification checks their hashes, but subsequent local
  tampering with a readable nonempty cached Parakeet file is not automatically
  repaired. Arbitrary explicit bundles and user dependency overrides are outside
  the qualification guarantee. Silero repair still requires network availability
  and upstream bytes matching the expected digest.
- Local C1/C2 stability, promotion checking, and C3 passed in the implementation
  package environment. The bootstrap C1/C2 aggregate assertions reported only
  the expected missing baseline; this was not a clean native gate verdict until
  C3. C1's scratch cache was cleaned before C2 reacquired assets, so this does not
  claim a warm C1/C2 cache-identity run. C2/C3 shared the later scratch cache.
- Local workflow structure, adversarial shell decisions, and offline plain-wheel
  origin/lifecycle checks have evidence. They are not full native installed-wheel
  release-mirror F3 evidence. Final F1-F4 verification and an authorized actual
  hosted PR/tag run have not yet been completed. Runner namespace availability,
  hosted performance, artifact transport, and publication remain unverified by
  hosted execution; no release or publication is claimed here.
- Evidence writing is best effort at real filesystem boundaries: unsafe output
  aliases are rejected before writing, and setup/I/O failures can leave no
  artifact. Per-file upload requirements and final aggregation still fail
  closed. A human-readable success message alone is never promotion evidence.

Implementation receipts are retained locally under
`.omo/evidence/backend-compatibility-envelope/` (tasks 9-12 and independent
verifier reports). They are intentionally not distributed or tracked. The
[qualified record](tests/integration/qualified_backend.json) is the committed
baseline, not a substitute for the outstanding verification above.
