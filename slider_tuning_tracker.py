import csv
import json
import time
from pathlib import Path


class SliderTuningTracker:
    """Tracks slider/range value changes across runs and writes ranking reports."""

    def __init__(self, report_dir: Path):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.report_dir / "slider_tuning_report.json"
        self.csv_path = self.report_dir / "slider_tuning_report.csv"
        self._stats: dict[str, dict] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.json_path.exists():
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats = data.get("stats", {})
            if isinstance(stats, dict):
                self._stats = stats
        except Exception:
            self._stats = {}

    def record_value(self, name: str, value: float) -> None:
        if not name:
            return
        key = name.strip()
        now = time.time()

        entry = self._stats.get(key)
        if entry is None:
            self._stats[key] = {
                "count": 1,
                "first_seen": now,
                "last_seen": now,
                "last_value": value,
                "min_value": value,
                "max_value": value,
                "sum_values": value,
                "sum_abs_delta": 0.0,
            }
            return

        prev = float(entry.get("last_value", value))
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_seen"] = now
        entry["last_value"] = value
        entry["min_value"] = min(float(entry.get("min_value", value)), value)
        entry["max_value"] = max(float(entry.get("max_value", value)), value)
        entry["sum_values"] = float(entry.get("sum_values", 0.0)) + value
        entry["sum_abs_delta"] = float(entry.get("sum_abs_delta", 0.0)) + abs(value - prev)

    def _rows(self) -> list[dict]:
        rows = []
        for name, entry in self._stats.items():
            count = max(1, int(entry.get("count", 0)))
            mean_value = float(entry.get("sum_values", 0.0)) / count
            rows.append(
                {
                    "name": name,
                    "count": count,
                    "last_value": float(entry.get("last_value", 0.0)),
                    "min_value": float(entry.get("min_value", 0.0)),
                    "max_value": float(entry.get("max_value", 0.0)),
                    "mean_value": mean_value,
                    "sum_abs_delta": float(entry.get("sum_abs_delta", 0.0)),
                    "first_seen": float(entry.get("first_seen", 0.0)),
                    "last_seen": float(entry.get("last_seen", 0.0)),
                }
            )
        rows.sort(key=lambda r: (r["count"], r["sum_abs_delta"]), reverse=True)
        return rows

    def save_reports(self) -> None:
        rows = self._rows()
        payload = {
            "generated_at": time.time(),
            "slider_count": len(rows),
            "stats": self._stats,
            "ranked": rows,
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "count",
                    "last_value",
                    "min_value",
                    "max_value",
                    "mean_value",
                    "sum_abs_delta",
                    "first_seen",
                    "last_seen",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
