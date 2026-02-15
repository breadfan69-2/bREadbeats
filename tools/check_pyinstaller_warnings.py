from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"^(missing|excluded) module named\s+(?:'([^']+)'|(\S+))\s+-\s+imported by\s+(.+)$"
)

DEFAULT_CRITICAL_PATTERNS = [
    r"^PyQt6$",
    r"^numpy$",
    r"^scipy$",
    r"^sounddevice$",
    r"^aubio$",
    r"^comtypes$",
    r"^pyaudiowpatch$",
    r"^pyqtgraph$",
]

DEFAULT_BENIGN_PATTERNS = [
    r"^pycparser\.(?:lextab|yacctab)$",
    r"^numpy\.f2py(?:\.|$)",
    r"^setuptools\._distutils$",
    r"^scipy\.special\._cdflib$",
    r"^org(?:\.python)?$",
    r"^java(?:\.lang)?$",
    r"^posix$",
    r"^pwd$",
    r"^resource$",
    r"^termios$",
    r"^_posix.*$",
    r"^_scproxy$",
    r"^_frozen_importlib(?:_external)?$",
    r"^pyimod02_importers$",
    r"^vms_lib$",
    r"^_winreg$",
    r"^threadpoolctl$",
    r"^numpy\._core\..+$",
    r"^multiprocessing\..+$",
    r"^ctypes\..+$",
]


def _compile_many(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def _matches_any(value: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def parse_warn_file(path: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = LINE_RE.match(line)
        if not match:
            continue
        status = match.group(1)
        module = match.group(2) or match.group(3)
        imported_by = match.group(4)
        findings.append(
            {
                "status": status,
                "module": module,
                "imported_by": imported_by,
                "line": line,
            }
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check PyInstaller warn file and fail only on critical missing modules."
    )
    parser.add_argument(
        "--warn-file",
        default="build/bREadbeats/warn-bREadbeats.txt",
        help="Path to PyInstaller warning report file.",
    )
    parser.add_argument(
        "--critical-pattern",
        action="append",
        default=[],
        help=(
            "Regex for module names that must not be missing. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--benign-pattern",
        action="append",
        default=[],
        help="Regex for known-benign modules to ignore. Can be passed multiple times.",
    )
    parser.add_argument(
        "--strict-unknown",
        action="store_true",
        help="Fail on unknown missing modules that are neither benign nor critical.",
    )

    args = parser.parse_args()
    warn_file = Path(args.warn_file)

    if not warn_file.exists():
        print(f"ERROR: warning file not found: {warn_file}")
        return 2

    critical_patterns = _compile_many(DEFAULT_CRITICAL_PATTERNS + args.critical_pattern)
    benign_patterns = _compile_many(DEFAULT_BENIGN_PATTERNS + args.benign_pattern)

    findings = parse_warn_file(warn_file)

    missing = [entry for entry in findings if entry["status"] == "missing"]
    excluded = [entry for entry in findings if entry["status"] == "excluded"]

    critical_hits: list[dict[str, str]] = []
    unknown_hits: list[dict[str, str]] = []
    benign_hits: list[dict[str, str]] = []

    for entry in missing:
        module = entry["module"]
        if _matches_any(module, benign_patterns):
            benign_hits.append(entry)
            continue
        if _matches_any(module, critical_patterns):
            critical_hits.append(entry)
            continue
        unknown_hits.append(entry)

    print(f"Checked: {warn_file}")
    print(
        f"Totals -> missing: {len(missing)}, excluded: {len(excluded)}, benign: {len(benign_hits)}, "
        f"critical: {len(critical_hits)}, unknown: {len(unknown_hits)}"
    )

    if critical_hits:
        print("\nCRITICAL missing modules:")
        for entry in critical_hits[:50]:
            print(f"- {entry['module']} (imported by {entry['imported_by']})")

    if args.strict_unknown and unknown_hits:
        print("\nUNKNOWN missing modules (strict mode):")
        for entry in unknown_hits[:50]:
            print(f"- {entry['module']} (imported by {entry['imported_by']})")

    if critical_hits:
        return 1
    if args.strict_unknown and unknown_hits:
        return 1

    print("Result: PASS (no critical missing modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
