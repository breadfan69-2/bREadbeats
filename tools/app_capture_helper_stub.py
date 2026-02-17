"""Application capture helper probe stub.

Contract:
- Invoked by audio_engine with:
  --probe --pid <int> --include-children <0|1>
- Emits one JSON object to stdout with at least:
  {"supported": bool, "reason": str}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import struct
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--include-children", type=int, choices=[0, 1], default=1)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--frames-per-buffer", type=int, default=1024)
    args = parser.parse_args()

    stream_supported = os.environ.get("BREADBEATS_APP_CAPTURE_STUB_STREAM", "0") == "1"

    if args.stream:
        if args.pid <= 0:
            return 2
        if not stream_supported:
            return 3

        frames = max(1, int(args.frames_per_buffer))
        channels = max(1, int(args.channels))
        sample_rate = max(8000, int(args.sample_rate))
        block = struct.pack('<' + ('f' * (frames * channels)), *([0.0] * (frames * channels)))
        sleep_s = frames / float(sample_rate)
        while True:
            try:
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
                time.sleep(sleep_s)
            except BrokenPipeError:
                return 0
            except Exception:
                return 1

    if not args.probe:
        print(json.dumps({"supported": False, "reason": "only --probe is implemented in stub"}))
        return 2

    if args.pid <= 0:
        print(json.dumps({"supported": False, "reason": "invalid pid"}))
        return 0

    if os.environ.get("BREADBEATS_APP_CAPTURE_STUB_SUPPORT", "0") == "1":
        print(json.dumps({
            "supported": True,
            "stream_enabled": stream_supported,
            "reason": "stub override enabled",
            "pid": args.pid,
            "include_children": bool(args.include_children),
        }))
        return 0

    print(json.dumps({
        "supported": False,
        "stream_enabled": False,
        "reason": "stub helper only; native app-capture backend not implemented",
        "pid": args.pid,
        "include_children": bool(args.include_children),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
