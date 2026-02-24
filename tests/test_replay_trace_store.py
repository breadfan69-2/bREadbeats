import gzip
import json
import os
import tempfile
import time
import unittest

from local_learning.replay_trace_store import (
    ReplayTraceStore,
    ReplayTraceStoreConfig,
    decimate_trace_rows,
)


class TestDecimateTraceRows(unittest.TestCase):
    def test_keeps_stride_and_event_rows(self):
        rows = [
            {"t": 0, "legacy_fire": False},
            {"t": 1, "legacy_fire": False},
            {"t": 2, "legacy_fire": True},
            {"t": 3, "legacy_fire": False},
            {"t": 4, "legacy_fire": False},
        ]
        kept = decimate_trace_rows(rows, keep_every=3)
        kept_t = [int(row["t"]) for row in kept]
        self.assertEqual(kept_t, [0, 2, 3])


class TestReplayTraceStore(unittest.TestCase):
    def test_write_trace_gzip_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ReplayTraceStore(
                ReplayTraceStoreConfig(root_dir=tmp_dir, compress=True)
            )
            out_path = store.write_trace(
                session_id="test_track",
                rows=[{"time": 1.0, "legacy_fire": True}, {"time": 1.1, "legacy_fire": False}],
            )
            self.assertTrue(out_path.name.endswith(".jsonl.gz"))
            self.assertTrue(out_path.exists())

            with gzip.open(out_path, mode="rt", encoding="utf-8") as handle:
                payload = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(payload), 2)
            self.assertTrue(payload[0]["legacy_fire"])

    def test_enforce_file_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ReplayTraceStore(
                ReplayTraceStoreConfig(
                    root_dir=tmp_dir,
                    max_files=2,
                    max_total_bytes=10_000_000,
                    max_age_days=365,
                    compress=False,
                )
            )
            for idx in range(3):
                store.write_trace(session_id=f"s_{idx}", rows=[{"n": idx}])
                time.sleep(0.01)

            usage = store.usage()
            self.assertEqual(usage["files"], 2)

    def test_enforce_total_size_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ReplayTraceStore(
                ReplayTraceStoreConfig(
                    root_dir=tmp_dir,
                    max_files=50,
                    max_total_bytes=900,
                    max_age_days=365,
                    compress=False,
                )
            )
            row = {"payload": "x" * 600}
            store.write_trace(session_id="a", rows=[row])
            time.sleep(0.01)
            store.write_trace(session_id="b", rows=[row])

            usage = store.usage()
            self.assertLessEqual(usage["total_bytes"], 900)
            self.assertEqual(usage["files"], 1)

    def test_enforce_age_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_path = os.path.join(tmp_dir, "old.jsonl")
            with open(old_path, "w", encoding="utf-8") as handle:
                handle.write("{\"x\":1}\n")

            stale_ts = time.time() - (10 * 24 * 3600)
            os.utime(old_path, (stale_ts, stale_ts))

            store = ReplayTraceStore(
                ReplayTraceStoreConfig(
                    root_dir=tmp_dir,
                    max_files=50,
                    max_total_bytes=100_000,
                    max_age_days=1,
                    compress=False,
                )
            )
            store.enforce_limits()

            self.assertFalse(os.path.exists(old_path))


if __name__ == "__main__":
    unittest.main()
