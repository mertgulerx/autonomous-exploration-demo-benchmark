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

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from benchmark_suite.backends import SyntheticBackend
from benchmark_suite.contracts import CampaignSpec, ContractError, canonical
from benchmark_suite.design import experiments
from benchmark_suite.engine import Engine
from benchmark_suite.library import Library
from benchmark_suite.native import NativeBackend


def specification():
    return {
        "name": "World limits",
        "robots": {"mogi": [{"id": "default-mogi", "name": "Default", "effective": {}}]},
        "worlds": {
            world: [{"id": world, "name": "Default", "x": 0.0, "y": 0.0, "yaw": 0.0}]
            for world in ("bookstore", "warehouse")
        },
        "world_time_limits": {"bookstore": 12.5, "warehouse": 0},
        "repetitions": 2,
        "concurrency": 2,
    }


class WorldLimitTests(unittest.TestCase):
    def test_limits_are_immutable_and_resolved_per_experiment(self):
        raw = specification()
        spec = CampaignSpec.accepted(raw)
        raw["world_time_limits"]["bookstore"] = 999
        self.assertEqual(
            spec.dictionary()["world_time_limits"], {"bookstore": 12.5, "warehouse": 0.0}
        )
        self.assertNotIn("time_limit", spec.dictionary())
        identity = str(uuid.uuid4())
        rows = list(experiments(spec, identity))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row.time_limit, 12.5 if row.world_id == "bookstore" else 0.0)
            self.assertEqual(row.dictionary()["time_limit"], row.time_limit)
        self.assertEqual(rows, list(experiments(spec, identity)))

    def test_legacy_budget_and_default_normalize_to_complete_world_limits(self):
        for value, expected in ((0, 0.0), (0.04, 0.04), (600, 600.0)):
            raw = specification()
            del raw["world_time_limits"]
            raw["time_limit"] = value
            spec = CampaignSpec.accepted(raw)
            self.assertEqual(
                dict(spec.world_time_limits), {"bookstore": expected, "warehouse": expected}
            )
            self.assertEqual(CampaignSpec.accepted(spec.dictionary()), spec)
        raw.pop("time_limit")
        self.assertEqual(
            dict(CampaignSpec.accepted(raw).world_time_limits),
            {"bookstore": 1800.0, "warehouse": 1800.0},
        )

    def test_invalid_budgets_are_rejected_before_any_worker_allocation(self):
        cases = [None, [], {}, {"bookstore": 1}, {"bookstore": 1, "warehouse": 2, "corridor": 3}]
        cases += [
            {"bookstore": value, "warehouse": 0}
            for value in (-1, True, "600", None, float("nan"), float("inf"))
        ]
        for limits in cases:
            with self.subTest(limits=limits):
                raw = specification()
                raw["world_time_limits"] = limits
                with self.assertRaises(ContractError):
                    CampaignSpec.accepted(raw)
        raw = specification()
        raw["time_limit"] = 99
        with self.assertRaises(ContractError):
            CampaignSpec.accepted(raw)
        with tempfile.TemporaryDirectory() as directory:
            backend = SyntheticBackend()
            engine = Engine(Path(directory), backend)
            try:
                with patch.object(backend, "start") as start:
                    response = engine.handle(
                        {
                            "protocol_version": 1,
                            "request_id": str(uuid.uuid4()),
                            "method": "campaign.submit",
                            "params": {"specification": raw},
                        }
                    )
                    self.assertFalse(response["ok"])
                    start.assert_not_called()
                    self.assertEqual(engine.store.rows("SELECT * FROM campaigns"), [])
            finally:
                engine.shutdown()

    def test_scheduler_preserves_world_limits_in_durable_attempt_configurations(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Path(directory), SyntheticBackend(steps=1))
            try:
                spec = CampaignSpec.accepted(specification())
                identity = engine.store.submit(spec)
                response = engine.handle(
                    {
                        "protocol_version": 1,
                        "request_id": str(uuid.uuid4()),
                        "method": "campaign.start",
                        "params": {"campaign_id": identity},
                    }
                )
                self.assertTrue(response["ok"], response)
                for _ in range(50):
                    engine.tick()
                attempts = engine.store.rows("SELECT * FROM attempts")
                self.assertEqual(len(attempts), 4)
                for attempt in attempts:
                    config = json.loads(attempt["configuration"])
                    self.assertEqual(
                        config["time_limit"], 12.5 if config["world_id"] == "bookstore" else 0.0
                    )
            finally:
                engine.shutdown()

    def test_native_launch_uses_snapshot_budget_and_supports_historical_experiments(self):
        root = Path(__file__).resolve().parents[2]
        library = Library(root, native=False)
        backend = NativeBackend.__new__(NativeBackend)
        backend.root, backend.profiles = root, {"default": {}}
        backend.runtime, backend.diagnostics = {"image": "unit-test-image"}, False
        profile = library.resolve_profile("mogi", {})
        with tempfile.TemporaryDirectory() as directory:
            attempt = {
                "id": "attempt",
                "slot": 0,
                "domain": 61,
                "token": "test-owner",
                "partition": "test-world-limits",
                "directory": str(Path(directory) / "attempts/attempt"),
            }
            for world, limit in (("bookstore", 12.5), ("warehouse", 0.0)):
                pose = next(
                    w["default_spawn"] for w in library.catalog()["worlds"] if w["id"] == world
                )
                configuration = {"profile": profile, "spawn": pose, "time_limit": limit}
                experiment = {
                    "world_id": world,
                    "robot_id": "mogi",
                    "id": "experiment",
                    "profile_id": "default-mogi",
                    "spawn_id": pose["id"],
                    "repetition": 0,
                    "specification": canonical(configuration),
                }
                campaign = {
                    "id": "campaign",
                    "specification": canonical({"worlds": {world: []}, "time_limit": 999}),
                }
                with patch("benchmark_suite.native.NativeWorker") as worker:
                    backend.start(attempt, experiment, campaign)
                    self.assertEqual(worker.call_args.args[1]["simulation_time_limit"], limit)
                del configuration["time_limit"]
                experiment["specification"] = canonical(configuration)
                for settings in ({"time_limit": 25.0}, {"world_time_limits": {world: limit}}):
                    campaign["specification"] = canonical(dict(settings, worlds={world: []}))
                    with patch("benchmark_suite.native.NativeWorker") as worker:
                        backend.start(attempt, deepcopy(experiment), campaign)
                        self.assertEqual(
                            worker.call_args.args[1]["simulation_time_limit"],
                            settings.get("time_limit", limit),
                        )
