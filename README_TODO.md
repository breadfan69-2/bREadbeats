# README TODO (Ongoing)

This file tracks ongoing README updates needed as behavior changes in the app.

## Open Items

- [ ] Document EXE logging behavior (release mode)
  - Default frozen EXE behavior: logs write to `breadbeats.log` in the same folder as `bREadbeats.exe`.
  - Default frozen EXE behavior: `stdout`/`stderr` are suppressed to avoid noisy `exe_startup_stdout.txt` / `exe_startup_stderr.txt` artifacts.
  - Debug override: set environment variable `BREADBEATS_DEBUG_STDIO=1` to keep stream output enabled for troubleshooting.
  - Add a short troubleshooting note explaining where to find startup/shutdown entries (`breadbeats.log`).

## Completed Items

- (none yet)
