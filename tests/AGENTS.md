# Test Suite Guidance

## OVERVIEW

This directory contains the deterministic root test suite, shared fixtures, and
repository or distribution contract tests. It covers unit-level behavior of the
shipped `sttx` modules without real model downloads or native inference.

`tests/e2e/` owns offline command and signal-process contracts. `tests/integration/`
owns tests that require current native packages, model assets, cache behavior, or
network access. Follow the child guidance when changing either scope.

## WHERE TO LOOK

| Test area | Test location | Production or support boundary |
| --- | --- | --- |
| CLI parsing, orchestration, streams, exits, diagnostics | `test_cli.py` | `sttx.cli`; inject `RunnerDependencies` and output seams |
| Audio conversion, failures, cleanup | `test_audio.py` | `sttx.audio` |
| VAD, decoding, timing, segments | `test_asr.py` | `sttx.asr`; use local fake recognizer and VAD types |
| Model resolution and cache rules | `test_model*.py` | `sttx.model`; use `model_helpers.py` |
| Output names, serialization, atomic writes | `test_output.py` | `sttx.output`; use `fixtures/output_transcript.json` for the golden schema |
| Wheel metadata and archive contents | `test_packaging.py` | `pyproject.toml` and the built wheel |
| Tracked repository boundary | `test_repo_contract.py` | git-tracked paths only |

`model_helpers.py` defines the exact model filenames, fake snapshot downloader,
asset writer, and bundle-path helper shared by root model tests. Keep its values
aligned with the production model contract rather than copying them into tests.

`conftest.py` registers integration evidence options and provides the optional
`write_identity_json` fixture. That writer creates parent directories, writes
sorted UTF-8 JSON through a staging file, fsyncs it, then atomically replaces the
requested evidence path.

## CONVENTIONS

- Test one public boundary or observable invariant at a time, using `tmp_path`
  for filesystem state and fakes for external dependencies.
- For CLI tests, inject a complete `RunnerDependencies` instance. Do not patch
  scattered production globals, and assert stdout, stderr, exit code, and call
  order where they are part of the contract.
- Keep model tests offline and deterministic with `model_helpers` fakes and
  temporary files. Root tests must not construct real recognizers or VADs.
- Treat `test_packaging.py` as an exact contract for the pure wheel name, entry
  point, metadata, inventory, and excluded local or model artifacts.
- Treat `test_repo_contract.py` as a tracked-path policy. Update its allow or
  deny lists only when the repository boundary deliberately changes.
- Keep fixtures small, hand-authored, and canonical. The sole shared fixture is
  `fixtures/output_transcript.json`, which fixes output JSON structure and UTF-8
  content for serializer tests.

## ANTI-PATTERNS

- Do not put end-to-end process assertions or real native and network probes in
  this directory. Move them to the applicable child test package.
- Do not add model or media binaries, wheels, caches, generated transcripts, or
  test evidence to tracked fixtures. Repository and wheel contracts reject them.
- Do not weaken exact assertions for package inventories, metadata, required
  files, forbidden paths, model filenames, atomic cleanup, or stream boundaries.
- Do not write identity evidence directly to its final path. Use the shared
  evidence writer when an integration test needs that optional artifact.
