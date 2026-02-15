import csv
import json
import time
from pathlib import Path


class AudioSessionReporter:
    """Persists per-session audio summaries to JSON and CSV reports."""

    def __init__(self, report_dir: Path, max_sessions: int = 200):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.report_dir / "audio_session_report.json"
        self.csv_path = self.report_dir / "audio_session_report.csv"
        self.max_sessions = max(1, int(max_sessions))

    def _load_existing_sessions(self) -> list[dict]:
        if not self.json_path.exists():
            return []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            sessions = payload.get("sessions", [])
            return sessions if isinstance(sessions, list) else []
        except Exception:
            return []

    def _to_builtin(self, value):
        if isinstance(value, dict):
            return {str(k): self._to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_builtin(v) for v in value]

        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return self._to_builtin(tolist())
            except Exception:
                pass

        item = getattr(value, "item", None)
        if callable(item):
            try:
                return self._to_builtin(item())
            except Exception:
                pass

        return value

    def save_session(self, session_summary: dict) -> None:
        sessions = self._load_existing_sessions()
        sessions.append(self._to_builtin(session_summary))
        if len(sessions) > self.max_sessions:
            sessions = sessions[-self.max_sessions :]

        payload = {
            "generated_at": time.time(),
            "session_count": len(sessions),
            "latest": sessions[-1],
            "sessions": sessions,
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        fieldnames = [
            "session_started_at",
            "session_ended_at",
            "seconds",
            "frames",
            "raw_rms_low",
            "raw_rms_high",
            "raw_rms_mean",
            "band_energy_low",
            "band_energy_high",
            "band_energy_mean",
            "flux_low",
            "flux_high",
            "flux_mean",
            "flux_high_threshold",
            "peak_high_threshold",
            "trough_low_threshold",
            "flux_high_total_s",
            "flux_high_episode_count",
            "flux_high_episode_mean_s",
            "flux_high_episode_max_s",
            "peak_high_total_s",
            "peak_high_episode_count",
            "peak_high_episode_mean_s",
            "peak_high_episode_max_s",
            "trough_low_total_s",
            "trough_low_episode_count",
            "trough_low_episode_mean_s",
            "trough_low_episode_max_s",
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sessions:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
