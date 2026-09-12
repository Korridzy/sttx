# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-04
**Commit:** c5c136e
**Branch:** 1-verbose-mode

## OVERVIEW

`sttx` is a Linux CPU transcription CLI. It normalizes one media file with
`ffmpeg`, resolves Parakeet and Silero assets, transcribes through
`sherpa-onnx`, then atomically writes JSON and plain-text transcripts.

## STRUCTURE

```text
.
├── src/sttx/          # Shipped flat Python package
├── tests/             # Unit, repository, and packaging contracts
│   ├── e2e/           # Offline CLI and signal-process contracts
│   ├── integration/   # Real native/model/network evidence gates
│   └── fixtures/      # Canonical output data
├── pyproject.toml     # PEP 621 metadata, entry point, tool configuration
└── README.md          # Public CLI, model, output, and installation contract
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI parsing, orchestration, logging, exits | `src/sttx/cli.py` | Public entry is `sttx.cli:main`; `RunnerDependencies` is the test seam |
| Media normalization and cleanup | `src/sttx/audio.py` | `ffmpeg` process group; 16 kHz mono signed 16-bit WAV |
| Model cache/download/offline bundle | `src/sttx/model.py` | Parakeet snapshot plus independently cached Silero VAD |
| VAD/decode/timing/segmentation | `src/sttx/asr.py` | Algorithmic core after normalized audio exists |
| Progress/debug event schema | `src/sttx/asr_events.py` | Closed activity-event union shared by ASR and CLI |
| Output naming and atomic writes | `src/sttx/output.py` | Sole final JSON/TXT filesystem boundary |
| Public behavior | `README.md`, `tests/e2e/` | Streams, exit codes, signals, silence, diagnostics |
| Distribution boundary | `tests/test_packaging.py` | Exact pure-wheel inventory and metadata |
| Allowed tracked layout | `tests/test_repo_contract.py` | Explicit required and forbidden paths/artifacts |
| Real dependency compatibility | `tests/integration/` | Native bindings, model assets, cache identity, evidence |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `main` / `run` | functions | `src/sttx/cli.py` | 38 | Console entry, signal handlers, error policy |
| `normalize_media` | function | `src/sttx/audio.py` | 22 | External `ffmpeg` boundary and temporary WAV lifecycle |
| `resolve_bundle` | function | `src/sttx/model.py` | 12 | Explicit/cache/network model resolution |
| `transcribe` | function | `src/sttx/asr.py` | 9 | VAD scan, decode, word timing, transcript assembly |
| `write_outputs` | function | `src/sttx/output.py` | 12 | Atomic two-file output replacement |
| `AsrActivity` | type union | `src/sttx/asr_events.py` | high fanout | Typed verbose/debug telemetry contract |

Runtime flow: `main` → `run` → `_run_production` → `_run` → input/output
validation → `normalize_media` → `resolve_bundle_cancellable` → recognizer/VAD
construction → `transcribe` → `write_outputs`.

## CONVENTIONS

- Python support is `>=3.11,<3.14`; Poetry creates the repository-local `.venv/`.
- Modules use `from __future__ import annotations` and strict Basedpyright types.
- Value objects and typed errors normally use `@dataclass(frozen=True, slots=True)`.
  Intentional mutable pipeline state is marked `# noqa: MUTABLE_OK`.
- Closed variants use `match` plus `assert_never`; keep activity events exhaustive.
- Stdout is reserved for the two final paths, JSON first and TXT second. All
  progress, warnings, errors, debug output, and cancellation text use stderr.
- Public exit codes are `0`, `1`, `2`, `130`, and `143`; preserve their meaning.
- JSON diagnostic lines have stable `event` and `elapsed_seconds` fields.
- Explicit `--model-dir` is zero-network and requires the exact flat five-file bundle.
- Output JSON is UTF-8, two-space indented, non-ASCII preserving, and newline
  terminated. TXT equals normalized transcript text with no trailing newline.
- Final outputs and test evidence use staging files, flush/fsync where specified,
  and `os.replace`; preserve cleanup on success, failure, and signals.

## ANTI-PATTERNS (THIS PROJECT)

- Do not track deploy/infra trees, `doc/`, model/media binaries, wheel artifacts,
  caches, generated transcriptions, or `poetry.lock`; the repository contract
  intentionally rejects them. The repository contract requires tracking
  `.github/dependabot.yml`, `.github/workflows/ci.yml`, and
  `.github/workflows/release.yml`; other `.github/` scaffolding needs a
  deliberate contract change.
- Do not add CI workflow scaffolding beyond the required `.github/` files without
  a deliberate repository contract change.
- Do not add a language option. Language comes from the model and falls back to `auto`.
- Do not acquire models before input and output validation succeeds.
- Do not weaken exact model filenames, warm-cache local-first behavior, output
  stem validation, atomic replacement, or signal cleanup guarantees.
- Do not let offline E2E tests open sockets or construct real recognizer/VAD objects.

## UNIQUE STYLES

- CLI tests inject a complete `RunnerDependencies` bundle rather than patching
  scattered production globals.
- E2E signal tests launch `tests/e2e/signal_driver.py` at named lifecycle phases.
- Real integration tests produce typed, hashed evidence and validate that evidence
  separately to reject false-positive runs.
- Oversized indivisible external probes carry an explicit `# noqa: SIZE_OK` reason.

## COMMANDS

```bash
poetry install
poetry check
poetry run basedpyright
poetry run pytest -q -m "not integration"
poetry run pytest -q
poetry run python -m compileall -q src tests
poetry run sttx --help
poetry run sttx --version
poetry build
```

## NOTES

- `ffmpeg` is a system prerequisite and is invoked even for already-compatible WAVs.
- The two `integration` tests use real native packages, model assets, network/cache
  behavior, process signals, and Linux tools; the unfiltered suite includes them.
- `sherpa_onnx` has no complete type stubs. Basedpyright may report third-party
  warnings, but first-party errors must remain zero.
- Packaging tests assert an exact wheel inventory; place agent guidance outside
  `src/sttx/` unless the distribution contract is deliberately updated.
