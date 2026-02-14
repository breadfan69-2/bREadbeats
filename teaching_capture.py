from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


def _to_native(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


class TeachingCapture:
    """In-memory teaching capture that flushes metrics/events to CSV and stores beat snapshots."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._lock = Lock()
        self.active = False
        self.session_dir: Path | None = None
        self.snapshots_dir: Path | None = None
        self.metrics_rows: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self._snapshot_counter = 0

    def start(self) -> Path:
        with self._lock:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.base_dir / "teaching_captures" / f"capture_{ts}"
            self.snapshots_dir = self.session_dir / "snapshots"
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_rows.clear()
            self.event_rows.clear()
            self._snapshot_counter = 0
            self.active = True
            return self.session_dir

    def stop(self, flush: bool = True) -> None:
        with self._lock:
            if not self.active:
                return
            if flush:
                self._flush_locked()
            self.active = False

    def add_metric(self, row: dict[str, Any]) -> None:
        with self._lock:
            if not self.active:
                return
            self.metrics_rows.append({k: _to_native(v) for k, v in row.items()})

    def add_event(self, row: dict[str, Any]) -> None:
        with self._lock:
            if not self.active:
                return
            self.event_rows.append({k: _to_native(v) for k, v in row.items()})

    def next_snapshot_path(self, prefix: str = "beat") -> Path | None:
        with self._lock:
            if not self.active or self.snapshots_dir is None:
                return None
            self._snapshot_counter += 1
            return self.snapshots_dir / f"{prefix}_{self._snapshot_counter:05d}.png"

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self.session_dir is None:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(self.session_dir / "metrics.csv", self.metrics_rows)
        self._write_csv(self.session_dir / "events.csv", self.event_rows)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            if not path.exists():
                path.write_text("", encoding="utf-8")
            return

        fieldnames: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
