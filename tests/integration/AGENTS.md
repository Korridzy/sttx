# Integration Test Guidance

## OVERVIEW

This directory holds the two `@pytest.mark.integration` evidence gates. They
exercise current native bindings, real model assets, and network behavior.

`test_binding_contract.py` is the native binding and VAD compatibility gate.
`test_real_pipeline.py` is the full CLI pipeline gate. Both produce evidence,
not mocked substitutes, and must keep their assertions tied to observed facts.

## WHERE TO LOOK

- `test_binding_contract.py` owns the indivisible binding probe and its typed
  evidence schema.
- `test_real_pipeline.py` owns the end-to-end gate, artifact lifecycle, and
  final evidence publication.
- `real_pipeline_runner.py` owns isolated environment setup and subprocess
  execution, including `strace` and `unshare` availability barriers.
- `real_pipeline_signals.py` and `real_pipeline_signal_process.py` own the
  SIGINT and SIGTERM probes at HF acquisition, Silero staging, and native decode.
- `real_pipeline_artifacts.py` owns paths, hashes, manifests, bundle copies,
  and atomic task evidence writes.
- `real_pipeline_evidence_contract.py` rejects malformed or false-positive
  full-pipeline evidence; update its focused tests with contract changes.

## CONVENTIONS

- Keep cold and warm cache roots isolated under the test temporary directory.
  Asset paths must remain inside the recorded isolated HF cache.
- Preserve `strace` and `unshare` barriers. They prove the tested subprocess
  crosses the real network and process boundaries, rather than merely invoking
  a local helper.
- Capture evidence as typed `JsonValue` mappings. Record asset paths, resolved
  paths, sizes, SHA-256 values, model repository and commit, and environment
  identity from the live run.
- Cold and warm evidence must have identical runtime asset identity for encoder,
  decoder, joiner, tokens, and Silero. Write identity output atomically through
  the existing staging, fsync, and `os.replace` paths.
- Signal probes must cover both signals at all three phases. HF requires live
  partial runtime data and missing finals, Silero requires a nonempty `.tmp`
  staging file, and native decode requires the child to announce entry into the
  native call and to record the time it stayed there, since a Python handler only
  runs once that call returns.
- Keep the explicit `# noqa: SIZE_OK` markers on the two gate modules. Each
  marks an indivisible external compatibility probe with helpers split out.

## ANTI-PATTERNS

- Don't replace real assets, native calls, downloads, tracing, or signals with
  mocks in either marked gate.
- Don't reuse a developer cache, accept bookkeeping files as HF partials, or
  publish evidence before `assert_todo10_contract` accepts it.
- Don't weaken cleanup receipts, post-signal manifest comparisons, or atomic
  evidence output to make flaky external behavior appear successful.
