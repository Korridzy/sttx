# SOURCE PACKAGE GUIDANCE

## OVERVIEW

`src/sttx` is the shipped, flat application package. Keep this guidance at
`src/` rather than inside `src/sttx/`: the wheel inventory treats package files
as distributed artifacts.

The package separates CLI policy, media preparation, model acquisition, ASR,
activity telemetry, and final output writes. Keep those boundaries explicit so
tests can substitute native and network dependencies without changing the CLI
contract.

## WHERE TO LOOK

| Change | Module | Ownership |
|---|---|---|
| Arguments, pipeline order, diagnostics, signals, exit translation | `sttx/cli.py` | Orchestration only. `RunnerDependencies` is the complete injection seam. |
| Temporary WAV creation and `ffmpeg` lifecycle | `sttx/audio.py` | Owns the child process group and `PreparedAudio` cleanup. |
| Local model validation, cache lookup, download staging | `sttx/model.py` | Produces `ModelBundle`; cancellable acquisition stays isolated here. |
| VAD traversal, recognizer decoding, timing, transcript assembly | `sttx/asr.py` | Defines the typed recognizer and VAD interaction protocols. |
| ASR-to-CLI progress vocabulary | `sttx/asr_events.py` | Owns the closed `AsrActivity` union and callback type. |
| Transcript values, output paths, serialization, replacement | `sttx/output.py` | The sole boundary for final transcript files. |
| Package version | `sttx/__init__.py` | Keep the public version value small and dependency-free. |

## CONVENTIONS

- Keep `cli.py` as a coordinator. Put native-library behavior in its owning
  module and add injectable callables to `RunnerDependencies` when the CLI
  needs to exercise that behavior in isolation.
- A `PreparedAudio` instance owns its temporary WAV. Consume it inside its
  context manager so success, exceptions, and cancellation all release it.
- `audio.py` must preserve the normalized WAV contract expected by ASR: mono,
  16 kHz, signed 16-bit PCM. Its `ffmpeg` child starts in a new session so
  cancellation can stop the whole process group.
- Model resolution returns validated paths only. Keep explicit-directory,
  local-cache, and acquisition decisions inside `model.py`; stage downloaded
  assets before replacement.
- Treat ASR protocols and `AsrActivity` variants as shared interfaces. When an
  activity variant changes, update producers and every exhaustive CLI consumer
  in the same change.
- Construct `Transcript` and `Segment` values before crossing into `output.py`.
  Keep output path parsing, text normalization, JSON shape, and replacement
  mechanics there rather than in the CLI or ASR code.

## ANTI-PATTERNS

- Do not bypass `RunnerDependencies` by patching module globals in CLI tests.
- Do not let ASR import CLI reporting code or emit untyped dictionaries for
  activity telemetry.
- Do not return a temporary WAV without its cleanup owner, or launch `ffmpeg`
  without the process-group shutdown path.
- Do not mix model network work into argument validation, decoding, or output
  writing.
- Do not write transcript files outside `output.py`, or replace final files
  without that module's staged-write cleanup path.
