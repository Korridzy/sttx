# sttx

`sttx` is a standalone Linux CPU transcription command. It normalizes one
audio/video file with `ffmpeg`, runs NVIDIA Parakeet TDT 0.6B v3 through the
`sherpa-onnx` Python bindings, and writes timestamped JSON and plain text
transcripts.

## Installation

### Prerequisite

Install `ffmpeg` separately and make sure it is available on `PATH`. The
application invokes it for every input, including WAV files that already have
the target format.

### Install the built wheel

The package produces a pure-Python wheel. Build it with Poetry, then install
that wheel into the target environment:

```bash
poetry build
python -m venv .venv-sttx
. .venv-sttx/bin/activate
python -m pip install dist/sttx-0.1.0-py3-none-any.whl
sttx --version
deactivate
```

The wheel installs the `sttx` console command and its runtime dependencies.

### Development environment

This repository keeps Poetry's virtual environment in `.venv/`:

```bash
poetry install
poetry env info --path
poetry check
poetry run pytest
poetry run python -m compileall -q src tests
poetry run sttx --help
poetry run sttx --version
poetry build
```

## Quickstart

```bash
poetry run sttx recording.mp4
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
sttx [-h] [-o OUTPUT] [-d OUTDIR] [--model-dir MODEL_DIR] [-v] [--debug] [--log-format {text,json}] [--version] media
```

- `media` — required positional path to one local audio/video file.
- `-h`, `--help` — show usage and exit.
- `-o OUTPUT`, `--output OUTPUT` — output filename stem. The program adds
  `.json` and `.txt`; a path or an existing suffix is rejected.
- `-d OUTDIR`, `--outdir OUTDIR` — output directory. Relative paths are
  resolved from the current directory; the default is `./transcriptions`.
- `--model-dir MODEL_DIR` — use a local, offline model bundle instead of
  acquiring model assets.
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
- `--version` — print `sttx 0.1.0` and exit.

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

Without `--model-dir`, the first run resolves the floating repository
`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` and downloads its
missing Parakeet assets into the Hugging Face cache. The Silero VAD asset is
stored in the local `sttx` cache. Model revision and download size are not
fixed promises and may change over time.

Warm-cache behavior is sticky and local-only first: an existing complete local
snapshot is used without a refresh request. Network acquisition is attempted
only when the local snapshot is missing or incomplete. The Silero VAD file is
also reused from its local `sttx` cache once present; it is not refreshed on a
normal warm run.

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

## Output files

The JSON file is UTF-8, uses two-space indentation, and keeps non-ASCII text
unescaped. Its `text` field is the segment text joined with single spaces.
`duration`, `start`, and `end` are rounded to two decimal places; segment IDs
start at zero.

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

The matching TXT file contains exactly the value of JSON `text`, with no
additional newline or other bytes:

```text
Hello world.
```

Silence/no speech is a successful result with `text: ""`, an empty
`segments` array, and `warning: no speech detected` on stderr.

## Attribution and licensing

`sttx` source code is licensed under the [Apache License 2.0](LICENSE).
Third-party software and model assets remain subject to their own terms; see
[NOTICE.md](NOTICE.md) for attribution and primary upstream license links.
