# E2E CLI and Signal Contracts

## OVERVIEW

This package proves the offline public CLI contract and cancellation behavior
without downloading models or constructing native ASR objects. Tests exercise
the CLI through injected dependencies or a separate Python subprocess.

`test_cli_contract.py` contains the direct CLI contract tests. It owns expected
streams, exit codes, output artifacts, and offline guard assertions.
`test_cli_signals.py` owns the process-level signal matrix. It starts
`signal_driver.py`, waits for a named barrier, then signals the child process.

`cli_contract_support.py` and `signal_driver.py` are support modules, not test
modules. Keep reusable fakes, guards, and controlled lifecycle barriers there;
keep pytest cases and their behavioral assertions in the `test_*.py` modules.

## WHERE TO LOOK

| Need | Location |
| --- | --- |
| Offline CLI fixtures, fakes, and guards | `cli_contract_support.py` |
| Direct CLI behavior and stream assertions | `test_cli_contract.py` |
| Named subprocess barriers and result capture | `signal_driver.py` |
| SIGINT and SIGTERM process assertions | `test_cli_signals.py` |

`install_runtime_guards()` must reject socket construction, real recognizer
creation, and real VAD creation. Tests must assert the guard counters stay at
zero when the contract requires an offline path.

`signal_driver.PHASES` names the lifecycle barriers: `ffmpeg`, `hf`, `silero`,
`decode`, `before_replace`, and `between_replaces`. Add a phase only when the
driver can prove it has reached the intended operation before the test signals it.

## CONVENTIONS

- Cover both `SIGINT` and `SIGTERM` for every signal-sensitive phase.
- Assert the exit is `128 + signum`, cancellation is reported on stderr, and
  stdout remains empty because it is reserved for successful output paths.
- Check cleanup results, including no staging files, no prepared audio, restored
  handlers, and termination of any recorded ffmpeg child process.
- Preserve the first-signal-wins rule during cleanup. A later signal must not
  replace the original cancellation reason or exit code.
- For replacement barriers, assert the documented final-file state: both old
  before the first replace, or only the allowed mixed pair between replaces.
- Use the driver's ready markers and bounded deadline. Don't replace them with
  timing sleeps that make process tests nondeterministic.

## ANTI-PATTERNS

- Don't open sockets, resolve remote models, or instantiate real recognizer or
  VAD objects in this package.
- Don't call `main()` directly when asserting signal delivery or child-process
  cleanup. Those contracts require the subprocess driver.
- Don't treat stderr as disposable diagnostic output. Assert its cancellation
  content and the absence of tracebacks.
- Don't weaken cleanup assertions to exit-code checks. A cancelled run must not
  leave staging artifacts, prepared audio, or a live ffmpeg process behind.
