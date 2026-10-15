# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import tempfile
import json
import threading
import time
import unittest
import uuid

from benchmark_suite.backends import SyntheticBackend
from benchmark_suite.contracts import CampaignSpec
from benchmark_suite.engine import Engine
from benchmark_suite.protocol import Client, MAX_MESSAGE, encode_response
from benchmark_suite.timing import TimingPolicy


def campaign(count=8, concurrency=2):
    return {
        "name": "Queue mechanics",
        "concurrency": concurrency,
        "repetitions": count,
        "robots": {
            "mogi": [{"id": "profile-a", "name": "Default", "effective": {}, "requested": {}}]
        },
        "worlds": {
            "warehouse": [{"id": "spawn-a", "name": "Default", "x": 1.0, "y": 1.0, "yaw": 0.0}]
        },
    }


def call(engine, method, params=None, request_id=None):
    response = engine.handle(
        {
            "protocol_version": 1,
            "request_id": request_id or str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
    )
    if not response["ok"]:
        raise AssertionError(response)
    return response["result"]


class EngineTests(unittest.TestCase):
    def test_attempt_cursor_does_not_repeat_rows_when_new_attempts_arrive(self):
        cid = call(self.engine, "campaign.submit", {"specification": campaign(8)})["campaign_id"]
        call(self.engine, "campaign.start", {"campaign_id": cid})
        for _ in range(100):
            self.engine.tick()
        self.engine.store.db.execute(
            "UPDATE attempts SET reserved_ns=1000 WHERE campaign_id=?", (cid,)
        )
        first = call(self.engine, "results.query", {"table": "attempts", "limit": 3})
        newer = call(self.engine, "campaign.submit", {"specification": campaign(1)})["campaign_id"]
        call(self.engine, "campaign.start", {"campaign_id": newer})
        self.engine.tick()
        second = call(
            self.engine,
            "results.query",
            {"table": "attempts", "limit": 3, "offset": 3, "cursor": first["next_cursor"]},
        )
        first_ids, second_ids = ({r["id"] for r in page["rows"]} for page in (first, second))
        self.assertFalse(first_ids & second_ids)
        previous = call(
            self.engine,
            "results.query",
            {"table": "attempts", "limit": 3, "cursor": second["previous_cursor"]},
        )
        self.assertEqual([r["id"] for r in previous["rows"]], [r["id"] for r in first["rows"]])
        last = call(
            self.engine,
            "results.query",
            {"table": "attempts", "limit": 3, "cursor": second["next_cursor"]},
        )
        self.assertEqual(len(first_ids | second_ids | {r["id"] for r in last["rows"]}), 8)
        self.assertIsNone(last["next_cursor"])

    def test_queue_reordering_is_durable_idempotent_and_keeps_active_campaign(self):
        ids = [
            call(self.engine, "campaign.submit", {"specification": campaign(1)})["campaign_id"]
            for _ in range(3)
        ]
        call(self.engine, "campaign.start", {"campaign_id": ids[0]})
        request = str(uuid.uuid4())
        params = {"campaign_id": ids[2], "direction": "first"}
        first = call(self.engine, "campaign.move", params, request)
        self.assertEqual(call(self.engine, "campaign.move", params, request), first)
        queued = self.engine.store.rows(
            "SELECT id FROM campaigns WHERE state='QUEUED' ORDER BY queue_position"
        )
        self.assertEqual([row["id"] for row in queued], [ids[2], ids[1]])
        self.assertEqual(self.engine.current, ids[0])
        with self.assertRaises(AssertionError):
            call(self.engine, "campaign.move", {"campaign_id": ids[0], "direction": "last"})
        for _ in range(30):
            self.engine.tick()
        draft = call(self.engine, "campaign.repeat", {"campaign_id": ids[0]})["specification"]
        self.assertEqual(draft["source_campaign_id"], ids[0])
        self.assertEqual(CampaignSpec.accepted(draft).count, 1)

    def test_report_result_integrity_failure_cannot_leave_a_running_report(self):
        from types import SimpleNamespace

        cid = call(self.engine, "campaign.submit", {"specification": campaign(1)})["campaign_id"]
        call(self.engine, "campaign.start", {"campaign_id": cid})
        for _ in range(20):
            self.engine.tick()
        rid = str(uuid.uuid4())
        directory = self.engine.workspace / "reports" / cid / rid
        directory.mkdir(parents=True)
        (directory / "partial.pdf").write_bytes(b"unreceipted")
        self.engine.store.db.execute(
            "INSERT INTO reports(id,campaign_id,state,specification,created_ns) VALUES(?,?,?,?,?)",
            (rid, cid, "RUNNING", "{}", time.time_ns()),
        )
        self.engine.store.db.execute(
            "UPDATE campaigns SET state='REPORTING',report_state='RUNNING' WHERE id=?", (cid,)
        )
        self.engine.current, self.engine.job_id, self.engine.job_stage = cid, rid, "report"
        self.engine.job = SimpleNamespace(
            poll=lambda: {
                "ok": True,
                "result": {
                    "index": str((directory / "missing.jsonl").relative_to(self.engine.workspace)),
                    "sha256": "0" * 64,
                },
            },
            close=lambda: None,
        )
        self.engine.tick()
        self.assertEqual(self.engine.store.campaign(cid)["state"], "FINISHED")
        self.assertEqual(self.engine.store.campaign(cid)["report_state"], "FAILED")
        self.assertEqual(
            self.engine.store.db.execute("SELECT state FROM reports WHERE id=?", (rid,)).fetchone()[
                0
            ],
            "FAILED",
        )
        self.assertFalse(directory.exists())
        self.assertIsNone(self.engine.job_id)
        self.assertIsNone(self.engine.current)
        self.assertIsNone(self.engine.job)

    def test_report_import_corruption_preserves_execution_and_cleans_outputs(self):
        cid = call(self.engine, "campaign.submit", {"specification": campaign(1)})["campaign_id"]
        call(self.engine, "campaign.start", {"campaign_id": cid})
        for _ in range(20):
            self.engine.tick()
        self.assertEqual(self.engine.store.campaign(cid)["state"], "FINISHED")
        execution_finished = self.engine.store.campaign(cid)["execution_finished_ns"]
        rid = str(uuid.uuid4())
        directory = self.engine.workspace / "reports" / cid / rid
        directory.mkdir(parents=True)
        (directory / "partial.pdf").write_bytes(b"incomplete")
        ledger = directory / "artifacts.jsonl"
        ledger.write_text(
            json.dumps(
                {
                    "path": str((directory / "missing.pdf").relative_to(self.engine.workspace)),
                    "sha256": "0" * 64,
                }
            )
            + "\n"
        )
        self.engine.store.db.execute(
            "INSERT INTO reports(id,campaign_id,state,specification,created_ns) VALUES(?,?,?,?,?)",
            (rid, cid, "INDEXING", "{}", time.time_ns()),
        )
        self.engine.store.db.execute(
            "UPDATE campaigns SET state='REPORTING',report_state='INDEXING' WHERE id=?", (cid,)
        )
        self.engine.current, self.engine.job_id = cid, rid
        self.engine.report_import = ledger.open()
        self.engine.tick()
        campaign_row = self.engine.store.campaign(cid)
        self.assertEqual(
            (campaign_row["state"], campaign_row["report_state"]), ("FINISHED", "FAILED")
        )
        self.assertEqual(campaign_row["execution_finished_ns"], execution_finished)
        self.assertFalse(directory.exists())
        self.assertIsNone(self.engine.current)
        self.assertEqual(
            self.engine.store.db.execute("SELECT state FROM reports WHERE id=?", (rid,)).fetchone()[
                0
            ],
            "FAILED",
        )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.engine = Engine(Path(self.directory.name), SyntheticBackend(steps=6))

    def tearDown(self):
        if self.engine is not None:
            self.engine.shutdown()
        self.directory.cleanup()

    def start(self, count=8, concurrency=2):
        identity = call(
            self.engine, "campaign.submit", {"specification": campaign(count, concurrency)}
        )["campaign_id"]
        call(self.engine, "campaign.start", {"campaign_id": identity})
        self.engine.tick()
        return identity

    def drain(self, maximum=300):
        peak = 0
        for _ in range(maximum):
            self.engine.tick()
            peak = max(peak, len(self.engine.active))
            if self.engine.current is None:
                return peak
        self.fail("Synthetic campaign did not finish")

    def test_mutation_deduplication_and_conflicting_content(self):
        request = str(uuid.uuid4())
        params = {"specification": campaign()}
        first = call(self.engine, "campaign.submit", params, request)
        self.assertEqual(call(self.engine, "campaign.submit", params, request), first)
        self.assertEqual(self.engine.store.query("campaigns", {})["total"], 1)
        params["specification"]["name"] = "Different"
        rejected = self.engine.handle(
            {
                "protocol_version": 1,
                "request_id": request,
                "method": "campaign.submit",
                "params": params,
            }
        )
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["error"]["code"], "conflict")
        self.assertFalse(self.engine.active)

    def test_pause_drains_slots_without_dispatch_then_resume(self):
        identity = self.start()
        call(self.engine, "campaign.pause", {"campaign_id": identity})
        for _ in range(20):
            self.engine.tick()
        self.assertEqual(self.engine.store.campaign(identity)["state"], "PAUSED")
        self.assertEqual(self.engine.store.counters(identity)["states"].get("DONE"), 2)
        self.assertFalse(self.engine.active)
        call(self.engine, "campaign.resume", {"campaign_id": identity})
        self.assertLessEqual(self.drain(), 2)
        self.assertEqual(self.engine.store.counters(identity)["states"], {"DONE": 8})

    def test_restart_discards_partial_files_and_uses_fresh_attempt(self):
        identity = self.start(count=1, concurrency=1)
        first_id = next(iter(self.engine.active))
        first = self.engine.store.attempt(first_id)
        partial = Path(first["directory"]) / "coverage.partial"
        partial.write_text("discard me")
        call(self.engine, "experiment.restart", {"experiment_id": first["experiment_id"]})
        self.engine.tick()
        self.assertFalse(partial.exists())
        self.assertTrue(self.engine.store.attempt(first_id)["discarded"])
        events = self.engine.store.rows("SELECT kind FROM events WHERE attempt_id=?", (first_id,))
        self.assertEqual(
            {row["kind"] for row in events}, {"attempt.discard_requested", "attempt.discarded"}
        )
        self.assertNotIn(first_id, self.engine.active)
        second = self.engine.store.attempt(next(iter(self.engine.active)))
        self.assertNotEqual(first["token"], second["token"])
        self.assertEqual(second["number"], 2)
        self.drain()
        self.assertEqual(self.engine.store.counters(identity)["attempts"]["discards"], 1)

    def test_discard_intent_survives_interrupted_cleanup(self):
        identity = self.start(count=1, concurrency=1)
        attempt = self.engine.store.attempt(next(iter(self.engine.active)))
        partial = Path(attempt["directory"]) / "path.partial"
        partial.write_text("discard me")
        call(self.engine, "experiment.stop", {"experiment_id": attempt["experiment_id"]})
        self.engine.store.close()
        self.engine = Engine(Path(self.directory.name), SyntheticBackend())
        self.assertFalse(partial.exists())
        self.assertEqual(self.engine.store.campaign(identity)["state"], "RECOVERY_REQUIRED")
        self.assertFalse(self.engine.active)
        self.assertEqual(self.engine.store.attempt(attempt["id"])["outcome"], "manual_discard")

    def test_unexpected_interruption_retains_partial_and_requires_resume(self):
        identity = self.start(count=1)
        attempt = self.engine.store.attempt(next(iter(self.engine.active)))
        partial = Path(attempt["directory"]) / "coverage.partial"
        partial.write_text("retain me")
        self.engine.store.close()
        self.engine = Engine(Path(self.directory.name), SyntheticBackend())
        self.assertTrue(partial.exists())
        self.assertEqual(self.engine.store.campaign(identity)["state"], "RECOVERY_REQUIRED")
        self.assertFalse(self.engine.active)
        call(self.engine, "campaign.resume", {"campaign_id": identity})
        self.drain()
        self.assertEqual(self.engine.store.counters(identity)["attempts"]["executed"], 2)

    def test_ordinary_failures_remain_outcomes(self):
        self.engine.backend = SyntheticBackend(lambda experiment, retry: "runtime_failure")
        identity = self.start(count=3, concurrency=4)
        self.drain()
        counters = self.engine.store.counters(identity)
        self.assertEqual(counters["outcomes"], {"runtime_failure": 3})
        self.assertEqual(counters["attempts"]["executed"], 3)

    def test_parallel_timing_rejections_get_one_sequential_replacement(self):
        self.engine.backend = SyntheticBackend(
            lambda experiment, retry: "completed" if retry else "timing_rejection"
        )
        identity = self.start(count=4, concurrency=2)
        self.drain()
        attempts = self.engine.store.query("attempts", {"campaign_id": identity}, limit=100)["rows"]
        self.assertEqual(len(attempts), 8)
        replacements = [a for a in attempts if a["retry"]]
        self.assertEqual(len(replacements), 4)
        self.assertTrue(all(a["slot"] == 0 for a in replacements))
        self.assertEqual(self.engine.store.counters(identity)["outcomes"], {"completed": 4})

    def test_second_timing_rejection_is_unresolved(self):
        self.engine.backend = SyntheticBackend(lambda experiment, retry: "timing_rejection")
        identity = self.start(count=2, concurrency=2)
        self.drain()
        self.assertEqual(self.engine.store.counters(identity)["states"], {"UNRESOLVED": 2})
        self.assertEqual(self.engine.store.counters(identity)["attempts"]["executed"], 4)

    def test_solo_attempt_does_not_get_redundant_timing_replacement(self):
        self.engine.backend = SyntheticBackend(lambda experiment, retry: "timing_rejection")
        identity = self.start(count=1, concurrency=4)
        self.drain()
        self.assertEqual(self.engine.store.counters(identity)["attempts"]["executed"], 1)
        self.assertEqual(self.engine.store.counters(identity)["states"], {"UNRESOLVED": 1})

    def test_invalid_query_filter_cannot_stop_running_workers(self):
        identity = self.start()
        response = self.engine.handle(
            {
                "protocol_version": 1,
                "request_id": str(uuid.uuid4()),
                "method": "results.query",
                "params": {"filters": {"campaign_id": []}},
            }
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["field"], "filters.campaign_id")
        self.assertEqual(self.engine.store.campaign(identity)["state"], "RUNNING")
        self.assertTrue(self.engine.active)

    def test_future_event_cursor_requires_snapshot_reset(self):
        response = call(self.engine, "events.read", {"after_sequence": 100})
        self.assertTrue(response["snapshot_required"])
        self.assertEqual(response["last_sequence"], 0)

    def test_outgoing_frame_size_is_bounded(self):
        encoded = encode_response({"ok": True, "result": "x" * MAX_MESSAGE})
        self.assertLess(len(encoded), MAX_MESSAGE)
        self.assertEqual(json.loads(encoded)["error"]["code"], "response_too_large")
        malformed = encode_response({"ok": False, "request_id": "x" * MAX_MESSAGE})
        self.assertLess(len(malformed), MAX_MESSAGE)
        self.assertIsNone(json.loads(malformed)["request_id"])

    def test_45000_database_experiments_are_paged_and_indexed(self):
        spec = CampaignSpec.accepted(campaign(count=45000))
        started = time.perf_counter()
        identity = self.engine.store.submit(spec)
        elapsed = time.perf_counter() - started
        started = time.perf_counter()
        page = self.engine.store.query(
            "experiments", {"campaign_id": identity}, offset=44000, limit=100
        )
        query_time = time.perf_counter() - started
        self.assertEqual(page["total"], 45000)
        self.assertEqual(len(page["rows"]), 100)
        self.assertLess(query_time, 1.0)
        print(f"\n45,000-row submission {elapsed:.3f}s; indexed page {query_time * 1000:.2f}ms")

    def test_unix_client_disconnect_leaves_engine_running(self):
        self.engine.shutdown()
        self.engine = None
        ready = threading.Event()
        holder = {}
        workspace = Path(self.directory.name)

        def serve():
            engine = Engine(workspace, SyntheticBackend())
            holder["engine"] = engine
            ready.set()
            engine.run()

        thread = threading.Thread(target=serve)
        thread.start()
        self.assertTrue(ready.wait(timeout=2))
        path = workspace / "runtime/control.sock"
        deadline = time.monotonic() + 2
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            client = Client(path)
            first = client.call("engine.status")
            self.assertEqual(first["backend"], "synthetic")
            self.assertTrue(thread.is_alive())
            self.assertEqual(Client(path).call("engine.status")["workspace"], str(workspace))
        finally:
            holder["engine"].closed = True
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())


class TimingTests(unittest.TestCase):
    def test_brief_drop_recovers_and_sustained_drop_rejects(self):
        policy = TimingPolicy()
        for wall in range(11):
            self.assertFalse(policy.update(float(wall), float(wall) * 0.8))
        for wall in range(11, 31):
            policy.update(float(wall), 8.0 + (wall - 10) * 1.2)
        self.assertFalse(policy.rejected)
        self.assertGreater(policy.brief_drops, 0)
        policy = TimingPolicy()
        for wall in range(16):
            policy.update(float(wall), float(wall) * 0.8)
        self.assertTrue(policy.rejected)

    def test_preparation_pause_cannot_reject(self):
        policy = TimingPolicy()
        for wall in range(100):
            policy.update(float(wall), 1.0, active=False)
        self.assertFalse(policy.rejected)

    def test_telemetry_gap_starts_a_new_window(self):
        policy = TimingPolicy()
        for wall in range(6):
            policy.update(float(wall), float(wall))
        policy.update(60.0, 6.0)
        self.assertIsNone(policy.rtf)
        self.assertFalse(policy.rejected)


if __name__ == "__main__":
    unittest.main()
