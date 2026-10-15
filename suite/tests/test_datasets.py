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

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from benchmark_suite.datasets import (
    compact,
    commit_compaction,
    compress_logs,
    files,
    materialize_manifest,
    observations,
    read_log,
    recover_checkpoints,
    recover_compactions,
    validate_attempt,
    supported_schema,
)
from benchmark_suite.schemas import SCHEMAS
from benchmark_suite.recording import Recorder
from benchmark_suite.storage import file_hash
from benchmark_suite.store import Store


def recorded(workspace, name, x):
    directory = workspace / "attempts" / name
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(
        campaign_id="campaign",
        experiment_id=name,
        attempt_id=name,
        robot_id="mogi",
        world_id="warehouse",
        profile_id="profile",
        spawn_id="spawn",
        repetition=1,
    )
    reference = SimpleNamespace(
        metadata={"metric_version": 1},
        metadata_bytes=b"{}",
        grid_bytes=b"grid",
        descriptor=lambda: {"id": "reference"},
    )
    recorder = Recorder(
        str(directory),
        dict(
            suite_identity=identity,
            suite_workspace=str(workspace),
            robot={"id": "mogi"},
            requested_parameters={},
            effective_parameters={},
        ),
        reference,
    )
    recorder.origins = 2.0, time.monotonic()
    recorder.clock = 3.0
    recorder.observation(
        "trajectory", dict(recorder.row("raw_odometry", 3_000_000_000), x=x, y=2.0)
    )
    recorder.finish(
        "time_limit",
        "Budget",
        {"motion_started": True, "coverage_status": "valid"},
        None,
        None,
        -1,
        [],
    )
    return directory


class DatasetTests(unittest.TestCase):
    def test_failed_attempt_without_coverage_alignment_still_has_a_report_summary(self):
        from benchmark_suite.analysis import DetectorSettings
        from benchmark_suite.report_data import summarize

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            directory = recorded(workspace, "failed", 1.0)
            target = workspace / "summary"
            target.mkdir()
            attempt = dict(
                id="failed",
                campaign_id="campaign",
                experiment_id="failed",
                outcome="startup_failure",
                reserved_ns=time.time_ns(),
            )
            experiment = dict(
                id="failed",
                robot_id="mogi",
                world_id="warehouse",
                profile_id="profile",
                spawn_id="spawn",
                repetition=0,
                block_id="block",
                state="DONE",
                final_attempt_id="failed",
                specification=json.dumps(
                    dict(profile=dict(name="Default"), spawn=dict(name="Default"))
                ),
            )
            summary = summarize(directory, attempt, experiment, DetectorSettings(), target)
            self.assertEqual(summary["outcome"], "startup_failure")
            self.assertIsNone(summary["completion_seconds"])
            self.assertTrue((target / "summary.json").is_file())

    def test_earlier_nullable_lateral_command_schema_remains_readable(self):
        import pyarrow as pa

        expected = SCHEMAS["trajectory"]
        earlier = expected.remove(expected.get_field_index("command_vy"))
        self.assertTrue(supported_schema("trajectory", earlier))
        wrong = earlier.set(earlier.get_field_index("vx"), pa.field("vx", pa.string()))
        self.assertFalse(supported_schema("trajectory", wrong))

    def test_compaction_requires_commit_preserves_rows_and_is_restartable(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            store = Store(workspace)
            try:
                first, second = (
                    recorded(workspace, "first", 1.0),
                    recorded(workspace, "second", 9.0),
                )
                source = files(first, "trajectory")[0]
                reference = compact(workspace, "campaign", [str(first), str(second)], 1024)
                self.assertTrue(source.exists())
                with self.assertRaisesRegex(OSError, "durably committed"):
                    commit_compaction(reference)
                index = workspace / reference["index"]
                with store.transaction():
                    store.db.execute(
                        "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            "index",
                            "campaign",
                            None,
                            "dataset_index",
                            str(index),
                            reference["sha256"],
                            index.stat().st_size,
                            None,
                            time.time_ns(),
                            "{}",
                        ),
                    )
                commit_compaction(reference)
                orphan = index.parent.parent / ("1" * 32)
                orphan.mkdir()
                (orphan / "interrupted.staging").write_bytes(b"uncommitted")
                recover_compactions(workspace)
                self.assertFalse(orphan.exists())
                self.assertTrue(index.exists())
                commit_compaction(reference)
                self.assertFalse(source.exists())
                self.assertEqual(observations(first, "trajectory")["x"].to_pylist(), [1.0])
                self.assertEqual(observations(second, "trajectory")["x"].to_pylist(), [9.0])
                validate_attempt(first)
                self.assertLess((first / "compacted.json").stat().st_size, 10000)
            finally:
                store.close()

    def test_recovery_rejects_changed_committed_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = recorded(Path(temporary), "attempt", 1.0)
            source = files(directory, "trajectory")[0]
            source.write_bytes(source.read_bytes() + b"unexpected")
            with self.assertRaises(OSError):
                validate_attempt(directory)
            result = recover_checkpoints(directory)
            self.assertTrue(result["rejected"])
            self.assertFalse(any(entry["kind"] == "trajectory" for entry in result["segments"]))

    def test_logs_remain_readable_at_byte_offsets_after_compression(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker.log"
            data = ("Controller started\nöğ\n" * 100000).encode()
            path.write_bytes(data)
            records = compress_logs(path.parent)
            self.assertFalse(path.exists())
            self.assertEqual(records[0]["sha256"], file_hash(path.with_suffix(".log.zst")))
            self.assertEqual(read_log(path, 23, 100), data[23:123])
            self.assertEqual(read_log(path, 1024**2 - 23, 100), data[1024**2 - 23 : 1024**2 + 77])
            self.assertEqual(read_log(path, len(data) + 5, 100), b"")

    def test_shared_configurations_are_verified_and_materialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            directory = recorded(workspace, "attempt", 1.0)
            raw = json.loads((directory / "manifest.json").read_bytes())
            self.assertEqual(set(raw["robot"]), {"object"})
            self.assertEqual(materialize_manifest(directory)["robot"]["id"], "mogi")
            digest = raw["robot"]["object"]
            (workspace / "objects" / digest[:2] / (digest + ".json")).write_text("{}")
            with self.assertRaisesRegex(OSError, "integrity"):
                materialize_manifest(directory)

    def test_interrupted_recording_recovers_only_readable_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = recorded(Path(temporary), "attempt", 1.0)
            source = files(directory, "trajectory")[0]
            (directory / "segments.json").unlink()
            manifest = json.loads((directory / "manifest.json").read_bytes())
            manifest.pop("segments_hash", None)
            manifest["phase"] = "RUNNING"
            (directory / "manifest.json").write_text(json.dumps(manifest))
            source.with_suffix(".staging").write_bytes(b"unfinished")
            recovered = recover_checkpoints(directory)
            self.assertTrue(recovered["truncated"])
            validate_attempt(directory)
            self.assertEqual(observations(directory, "trajectory")["x"].to_pylist(), [1.0])

    def test_engine_recovery_registers_partial_artifacts_and_keeps_not_done(self):
        from benchmark_suite.backends import SyntheticBackend
        from benchmark_suite.contracts import CampaignSpec
        from benchmark_suite.engine import Engine
        from test_engine import call, campaign

        class RecoveryBackend(SyntheticBackend):
            synthetic = False

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            engine = Engine(workspace, SyntheticBackend())
            cid = call(engine, "campaign.submit", {"specification": campaign(1, 1)})["campaign_id"]
            call(engine, "campaign.start", {"campaign_id": cid})
            engine.tick()
            aid = next(iter(engine.active))
            directory = recorded(workspace, aid, 1.0)
            manifest = json.loads((directory / "manifest.json").read_bytes())
            manifest["phase"] = "RUNNING"
            (directory / "manifest.json").write_text(json.dumps(manifest))
            engine.store.close()
            recovered = Engine(workspace, RecoveryBackend(), CampaignSpec.accepted)
            try:
                manifest = materialize_manifest(directory)
                self.assertTrue(manifest["partial"])
                self.assertEqual(manifest["outcome"], "engine_interruption")
                self.assertEqual(recovered.store.counters(cid)["states"], {"NOT_DONE": 1})
                receipts = recovered.store.rows(
                    "SELECT kind FROM artifacts WHERE attempt_id=?", (aid,)
                )
                self.assertTrue(any(row["kind"] == "manifest" for row in receipts))
                self.assertTrue(any(row["kind"] == "checkpoint_index" for row in receipts))
                self.assertEqual(len(observations(directory, "trajectory")), 1)
            finally:
                recovered.shutdown()

    def test_structurally_invalid_manifest_does_not_abort_recovery(self):
        from benchmark_suite.backends import SyntheticBackend
        from benchmark_suite.engine import Engine
        from test_engine import call, campaign

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            engine = Engine(workspace, SyntheticBackend())
            cid = call(engine, "campaign.submit", {"specification": campaign(1, 1)})["campaign_id"]
            call(engine, "campaign.start", {"campaign_id": cid})
            engine.tick()
            aid = next(iter(engine.active))
            directory = Path(engine.store.attempt(aid)["directory"])
            (directory / "manifest.json").write_text("[]")
            engine.store.close()
            recovered = Engine(workspace, SyntheticBackend())
            try:
                self.assertEqual(recovered.store.attempt(aid)["outcome"], "engine_interruption")
                self.assertEqual(
                    json.loads((directory / "manifest.json").read_bytes())["phase"], "FINISHED"
                )
            finally:
                recovered.shutdown()


if __name__ == "__main__":
    unittest.main()
