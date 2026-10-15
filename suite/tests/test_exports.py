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
import uuid

from benchmark_suite.backends import SyntheticBackend
from benchmark_suite.datasets import compact, commit_compaction, observations
from benchmark_suite.engine import Engine
from benchmark_suite.exports import copy_export, export_campaign, verify_export
from benchmark_suite.recording import Recorder
from benchmark_suite.storage import file_hash


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "active"
        self.engine = Engine(self.workspace, SyntheticBackend(steps=1))
        design = {
            "name": "Portable dataset",
            "concurrency": 1,
            "repetitions": 1,
            "robots": {
                "mogi": [{"id": "profile", "name": "Default", "requested": {}, "effective": {}}]
            },
            "worlds": {
                "warehouse": [{"id": "spawn", "name": "Default", "x": 1.0, "y": 1.0, "yaw": 0.0}]
            },
        }
        self.campaign = self.call("campaign.submit", {"specification": design})["campaign_id"]
        self.call("campaign.start", {"campaign_id": self.campaign})
        for _ in range(10):
            self.engine.tick()
        self.attempt = self.engine.store.one("SELECT * FROM attempts")

    def tearDown(self):
        self.engine.shutdown()
        self.temporary.cleanup()

    def request(self, method, params, request_id=None):
        return self.engine.handle(
            dict(
                protocol_version=1,
                request_id=request_id or str(uuid.uuid4()),
                method=method,
                params=params,
            )
        )

    def call(self, method, params, request_id=None):
        response = self.request(method, params, request_id)
        self.assertTrue(response["ok"], response)
        return response["result"]

    def payload(self):
        return dict(
            workspace=str(self.workspace), campaign_id=self.campaign, export_id=str(uuid.uuid4())
        )

    def test_export_is_durable_idempotent_and_blocks_new_execution(self):
        identity = str(uuid.uuid4())
        result = self.call("dataset.export", {"campaign_id": self.campaign}, identity)
        self.assertEqual(
            result, self.call("dataset.export", {"campaign_id": self.campaign}, identity)
        )
        conflict = self.request("campaign.resume", {"campaign_id": self.campaign})
        self.assertFalse(conflict["ok"])
        deadline = time.monotonic() + 15
        while self.engine.exports.identity and time.monotonic() < deadline:
            self.engine.tick()
            time.sleep(0.02)
        entry = self.call("exports.get", {"export_id": result["export_id"]})
        self.assertEqual(entry["state"], "FINISHED", entry)
        manifest = verify_export(Path(entry["path"]))
        self.assertEqual(manifest["campaign_id"], self.campaign)
        self.assertEqual(self.engine.store.one("SELECT COUNT(*) n FROM exports")["n"], 1)

    def test_corrupt_or_missing_indexed_source_cannot_be_exported(self):
        path = Path(self.attempt["directory"]) / "manifest.json"
        original = path.read_bytes()
        for corrupt in (True, False):
            if corrupt:
                path.write_bytes(original + b" ")
            else:
                path.unlink()
            payload = self.payload()
            with self.assertRaises(OSError):
                export_campaign(payload)
            self.assertFalse((self.workspace / "exports" / payload["export_id"]).exists())
            self.assertFalse(
                (self.workspace / "exports" / (payload["export_id"] + ".staging")).exists()
            )
            path.write_bytes(original)

    def test_compacted_data_can_be_read_after_bundle_relocation(self):
        directory = Path(self.attempt["directory"])
        reference = SimpleNamespace(
            metadata={"metric_version": 1},
            metadata_bytes=b"{}",
            grid_bytes=b"grid",
            descriptor=lambda: {"id": "ref"},
        )
        identity = {key: self.attempt[key] for key in ("campaign_id", "experiment_id")}
        identity.update(
            attempt_id=self.attempt["id"],
            robot_id="mogi",
            world_id="warehouse",
            profile_id="profile",
            spawn_id="spawn",
            repetition=0,
        )
        recorder = Recorder(
            str(directory),
            dict(
                suite_identity=identity,
                suite_workspace=str(self.workspace),
                robot={"id": "mogi"},
                requested_parameters={},
                effective_parameters={},
            ),
            reference,
        )
        recorder.origins = 1.0, time.monotonic()
        recorder.clock = 2.0
        recorder.observation(
            "trajectory", dict(recorder.row("raw_odometry", 2_000_000_000), x=1.25, y=2.0)
        )
        recorder.finish("time_limit", "Budget", {"motion_started": True}, None, None, -1, [])
        self.engine.store.db.execute(
            "UPDATE artifacts SET hash=? WHERE attempt_id=? AND kind='manifest'",
            (file_hash(directory / "manifest.json"), self.attempt["id"]),
        )
        receipt = compact(self.workspace, self.campaign, [str(directory)])
        self.engine.store.db.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                self.campaign,
                None,
                "dataset_index",
                str(self.workspace / receipt["index"]),
                receipt["sha256"],
                (self.workspace / receipt["index"]).stat().st_size,
                None,
                time.time_ns(),
                "{}",
            ),
        )
        commit_compaction(receipt)
        for entry in json.loads((self.workspace / receipt["index"]).read_bytes())["shards"]:
            self.engine.store.db.execute(
                "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    self.campaign,
                    None,
                    "dataset",
                    str(self.workspace / entry["path"]),
                    entry["sha256"],
                    entry["bytes"],
                    entry["rows"],
                    time.time_ns(),
                    "{}",
                ),
            )
        bundle = Path(export_campaign(self.payload())["path"])
        destination = Path(self.temporary.name) / "relocated"
        copy_export(bundle, destination)
        verify_export(destination)
        table = observations(destination / "attempts" / self.attempt["id"], "trajectory")
        self.assertEqual(table["x"].to_pylist(), [1.25])

    def test_cancellation_and_corruption_remove_only_owned_copy_staging(self):
        bundle = Path(export_campaign(self.payload())["path"])
        destination = Path(self.temporary.name) / "copy"
        with self.assertRaises(InterruptedError):
            copy_export(bundle, destination, cancelled=lambda: True)
        self.assertFalse(destination.exists())
        self.assertFalse(list(destination.parent.glob("copy.*.staging")))
        self.assertTrue((bundle / "export.json").is_file())
        with self.assertRaises(ValueError):
            copy_export(bundle, bundle.parent)
        self.assertTrue(bundle.is_dir())
        (bundle / "manifest.json").write_text("extra")
        index = json.loads((bundle / "export.json").read_bytes())
        first = json.loads((bundle / index["index"]).read_text().splitlines()[0])
        (bundle / first["path"]).write_text("corrupt")
        with self.assertRaises(OSError):
            verify_export(bundle)
