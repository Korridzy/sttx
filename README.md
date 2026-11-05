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
Linux namespace support. `poetry.lock` remains ignored local state.

The [repository contract](tests/test_repo_contract.py) defines the approved
tracked layout. The [packaging contract](tests/test_packaging.py) checks the
exact pure-wheel inventory and dependency metadata. Models, media, run evidence,
wheels, caches, and lockfiles are not repository deliverables.

CI requires additional qualification when the speech backend or its committed
qualification record changes. See [KNOWN-GAPS.md](KNOWN-GAPS.md) for the limits
of that coverage.

## Attribution and licensing

`sttx` source code is licensed under the [GNU General Public License v3.0 only](LICENSE).
Third-party software and model assets remain subject to their own terms; see
[NOTICE.md](NOTICE.md) for attribution and primary upstream license links.
