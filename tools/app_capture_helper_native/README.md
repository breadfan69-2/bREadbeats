# app_capture_helper_native

Native Windows helper scaffold for per-process loopback capture.

This folder is a **starter skeleton** that matches the Python-side helper contract used by `audio_engine.py`.
It currently returns `supported=false` and does not capture real audio yet.

## Build

```powershell
cmake -S tools/app_capture_helper_native -B tools/app_capture_helper_native/build -G "Visual Studio 17 2022"
cmake --build tools/app_capture_helper_native/build --config Release
```

Expected output binary:

- `tools/app_capture_helper_native/build/Release/app_capture_helper_native.exe`

## Contract

The helper should support:

- `--probe --pid <int> --include-children <0|1>`
  - stdout: JSON object with keys:
    - `supported` (bool)
    - `stream_enabled` (bool)
    - `reason` (string)

- `--stream --pid <int> --include-children <0|1> --sample-rate <int> --channels <int> --frames-per-buffer <int>`
  - stdout: raw interleaved `float32` PCM frames, little-endian
  - no framing bytes; Python side reads fixed-size frame blocks

See `PROTOCOL.md` for more detail.
