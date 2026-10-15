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
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from benchmark_suite.contracts import ContractError
from benchmark_suite.library import Library, sweep_values
from benchmark_suite.store import Store
from exploration_benchmark_robots.preflight import validate_spawn

ROOT = Path(__file__).resolve().parents[2]


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = Library(ROOT, native=False)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.directory.name))
        self.library.bind(self.store)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def operation(self, method, params):
        with self.store.transaction():
            return self.library.handle(method, params, self.store)

    def test_robot_defaults_preserve_mogi_baseline_and_native_scoring(self):
        default = self.library.resolve_profile("mogi", {})
        self.assertEqual(default["effective"], self.library.schema.defaults)
        self.assertEqual(
            default["robot"]["benchmark_settings"]["effective"],
            {"movement_speed": 0.8, "turning_speed": 1.0, "lidar_range": 10.0},
        )
        native = self.library.resolve_profile("tb3_burger", {})
        self.assertEqual(
            native["effective"]["max_linear_speed_vmax"], native["robot"]["max_linear_speed"]
        )
        explicit = self.library.resolve_profile(
            "tb3_burger", {"requested": {"max_linear_speed_vmax": 0.7}}
        )
        self.assertEqual(explicit["effective"]["max_linear_speed_vmax"], 0.7)
        self.assertEqual(self.library.schema.defaults, default["effective"])

    def test_revisions_are_immutable_and_exports_preserve_types(self):
        original = self.operation(
            "profiles.save", {"robot_id": "mogi", "name": "Baseline", "tags": ["research"]}
        )
        edited = self.operation(
            "profiles.save",
            {
                "robot_id": "mogi",
                "name": "Fast",
                "family_id": original["family_id"],
                "requested": {"max_linear_speed_vmax": 0.6},
            },
        )
        self.assertEqual(edited["revision"], 2)
        self.assertEqual(
            self.library._revision(original["id"], "profile")["effective"], original["effective"]
        )
        exported = self.operation("profiles.export", {"revision_id": edited["id"]})["yaml"]
        imported = self.library.resolve_profile("mogi", {"yaml_text": exported})
        self.assertEqual(imported["effective"], edited["effective"])
        self.assertEqual(
            {k: type(v) for k, v in imported["effective"].items()},
            {k: type(v) for k, v in edited["effective"].items()},
        )
        complete = self.operation(
            "profiles.export", {"revision_id": edited["id"], "format": "suite"}
        )
        clone = self.operation("profiles.import", {"yaml_text": complete["yaml"]})
        self.assertEqual(clone["effective"], edited["effective"])
        self.assertEqual(
            self.operation("profiles.list", {"search": "Baseline", "tag": "research"})["total"], 1
        )

    def test_yaml_and_unknown_settings_are_rejected(self):
        for text in (
            "frontier_explorer:\n  ros__parameters:\n    autostart: true\n    autostart: false\n",
            "bad_node:\n  ros__parameters: {}\n",
            "frontier_explorer:\n  ros__parameters:\n    max_linear_speed_vmax: .nan\n",
        ):
            with self.assertRaises(ContractError):
                self.library.resolve_profile("mogi", {"yaml_text": text})
        with self.assertRaises(ContractError):
            self.library.resolve_profile("mogi", {"unused": 1})
        with self.assertRaises(ContractError):
            self.library.resolve_profile("mogi", {"requested": {"use_sim_time": False}})
        with self.assertRaises(ContractError):
            self.operation("profiles.save", {"robot_id": "mogi", "family_id": []})

    def test_decimal_sweeps_and_physical_differences(self):
        self.assertEqual(
            sweep_values({"start": 0.1, "stop": 0.3, "step": 0.1}, 0.0), [0.1, 0.2, 0.3]
        )
        self.assertEqual(sweep_values({"start": 3, "stop": 1, "step": -1}, 0), [3, 2, 1])
        for axis in (
            {"values": [1, 1]},
            {"start": 1, "stop": 3, "step": -0.1},
            {"start": 1, "stop": 2, "step": 0.5},
        ):
            with self.assertRaises(ContractError):
                sweep_values(axis, 0)
        variants = self.operation(
            "profiles.sweep",
            {
                "robot_id": "mogi",
                "axes": {
                    "max_linear_speed_vmax": {"values": [0.4, 0.8]},
                    "robot.lidar_range": {"values": [8.0, 10.0]},
                },
            },
        )
        self.assertEqual(variants["count"], 4)
        self.assertFalse(variants["profiles"])
        diff = self.operation(
            "profiles.diff",
            {"left": variants["revision_ids"][0], "right": variants["revision_ids"][1]},
        )
        self.assertFalse(diff["algorithm"])
        self.assertEqual(diff["physical"]["lidar_range"], [8.0, 10.0])
        accepted = self.library.resolve_campaign(
            {
                "name": "Sweep provenance",
                "robots": {"mogi": variants["revision_ids"]},
                "worlds": {"bookstore": ["Default"]},
            }
        )
        profiles = accepted.dictionary()["robots"]["mogi"]
        self.assertEqual(len(profiles), 4)
        self.assertEqual(profiles[1]["sweep"]["values"]["robot.lidar_range"], 10.0)
        from benchmark_suite.factor_reports import slices

        self.assertEqual(len(slices(accepted.dictionary())), 12)
        altered = dict(profiles[0])
        altered["requested"] = {**altered["requested"], "max_linear_speed_vmax": 0.7}
        with self.assertRaisesRegex(ContractError, "provenance"):
            self.library.resolve_profile("mogi", altered)
        with self.assertRaises(ContractError):
            self.operation(
                "profiles.sweep_preview",
                {"robot_id": "mogi", "axes": {"use_sim_time": {"values": [True]}}},
            )

    def test_custom_spawns_keep_reference_targets_and_reject_collisions(self):
        for world, ref in self.library.references.items():
            before = ref.target.copy()
            x, y, _, yaw = self.library.scenarios[world]["spawn"]
            saved = self.operation(
                "spawns.save", {"world_id": world, "name": "Custom", "x": x, "y": y, "yaw": yaw}
            )
            spec = self.library.resolve_campaign(
                {"name": "Spawn", "robots": {"mogi": ["Default"]}, "worlds": {world: [saved["id"]]}}
            )
            self.assertEqual(spec.worlds[0][1][0].yaw, yaw)
            np.testing.assert_array_equal(before, ref.target)
            with self.assertRaises(ContractError):
                self.operation(
                    "spawns.save",
                    {"world_id": world, "name": "Outside", "x": 1e5, "y": 1e5, "yaw": 0.0},
                )
        with self.assertRaises(ContractError):
            self.library.resolve_campaign(
                {
                    "name": "Invalid",
                    "robots": {"mogi": ["Default"]},
                    "worlds": {"missing": ["Default"]},
                }
            )

    def test_rotated_footprint_does_not_use_empty_bounding_corners(self):
        collision = np.zeros((20, 20), dtype=bool)
        collision[5, 10] = True
        ref = SimpleNamespace(
            origin=[0.0, 0.0, 0.0],
            resolution=1.0,
            collision=collision,
            inclusion=np.ones((20, 20), dtype=bool),
        )
        robot = {
            "label": "Rectangle",
            "footprint": [[-4.0, -0.2], [4.0, -0.2], [4.0, 0.2], [-4.0, 0.2]],
        }
        validate_spawn(robot, [8.0, 8.0, 0.0, np.pi / 4], ref)
        collision[8, 8] = True
        with self.assertRaises(ValueError):
            validate_spawn(robot, [8.0, 8.0, 0.0, np.pi / 4], ref)


if __name__ == "__main__":
    unittest.main()
