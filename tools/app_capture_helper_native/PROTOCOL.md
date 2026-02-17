# Helper Protocol (Native)

## Probe mode

Command:

```text
app_capture_helper_native --probe --pid 1234 --include-children 1
```

Output to stdout (single JSON line):

```json
{"supported":false,"stream_enabled":false,"reason":"not implemented"}
```

## Stream mode

Command:

```text
app_capture_helper_native --stream --pid 1234 --include-children 1 --sample-rate 44100 --channels 2 --frames-per-buffer 1024
```

Output:

- Raw PCM `float32` little-endian, interleaved channels, exactly `frames-per-buffer * channels * 4` bytes per block.
- Emit continuously until process exits or pipe closes.

## Error signaling

- Return non-zero exit code for fatal launch/runtime errors.
- Write human-readable diagnostics to stderr.
- Keep stdout reserved for probe JSON or stream PCM.

## Future enhancements

- Optional metadata sidechannel (named pipe) for timing and underrun metrics.
- Runtime resampling option if endpoint format differs from requested format.
