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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--include-children", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    if not args.probe:
        print(json.dumps({"supported": False, "reason": "only --probe is implemented in stub"}))
        return 2

    if args.pid <= 0:
        print(json.dumps({"supported": False, "reason": "invalid pid"}))
        return 0

    if os.environ.get("BREADBEATS_APP_CAPTURE_STUB_SUPPORT", "0") == "1":
        print(json.dumps({
            "supported": True,
            "reason": "stub override enabled",
            "pid": args.pid,
            "include_children": bool(args.include_children),
        }))
        return 0

    print(json.dumps({
        "supported": False,
        "reason": "stub helper only; native app-capture backend not implemented",
        "pid": args.pid,
        "include_children": bool(args.include_children),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
