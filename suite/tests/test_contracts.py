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

from dataclasses import FrozenInstanceError
import unittest
import uuid

from benchmark_suite.contracts import CampaignSpec, ContractError
from benchmark_suite.design import experiments
from exploration_benchmark_runner.launch_spec import WorkerLaunchSpec


def design(robots=2, profiles=3, worlds=2, spawns=2, repetitions=3):
    return {
        "name": "Parameter experiment",
        "repetitions": repetitions,
        "robots": {
            f"robot-{r}": [
                {
                    "id": f"profile-{r}-{p}",
                    "name": f"Profile {p}",
                    "requested": {"weight": float(p)},
                    "effective": {"weight": float(p)},
                }
                for p in range(profiles)
            ]
            for r in range(robots)
        },
        "worlds": {
            f"world-{w}": [
                {"id": f"spawn-{w}-{s}", "name": f"Spawn {s}", "x": float(s), "y": 0.0, "yaw": 1.57}
                for s in range(spawns)
            ]
            for w in range(worlds)
        },
    }


class ContractTests(unittest.TestCase):
    def test_snapshot_does_not_share_mutable_parameters(self):
        raw = design()
        accepted = CampaignSpec.accepted(raw)
        raw["robots"]["robot-0"][0]["effective"]["weight"] = 9.0
        self.assertEqual(accepted.dictionary()["robots"]["robot-0"][0]["effective"]["weight"], 0.0)
        with self.assertRaises(FrozenInstanceError):
            accepted.limit = 10

    def test_count_handles_nonuniform_profiles_and_explicit_exclusions(self):
        raw = design()
        raw["robots"]["robot-0"].pop()
        raw["exclusions"] = [["robot-1", "world-0", "spawn-0-0"]]
        spec = CampaignSpec.accepted(raw)
        self.assertEqual(spec.count, (2 * 4 + 3 * 3) * 3)

    def test_schedule_is_deterministic_and_rotates_profiles(self):
        spec = CampaignSpec.accepted(design(robots=1, worlds=1, spawns=1))
        identity = str(uuid.uuid4())
        first = list(experiments(spec, identity))
        self.assertEqual(first, list(experiments(spec, identity)))
        self.assertEqual([r.position for r in first], list(range(spec.count)))
        for repetition in range(3):
            block = [r for r in first if r.repetition == repetition]
            self.assertEqual(block[0].profile.id, f"profile-0-{repetition}")
        self.assertEqual(len({r.id for r in first}), spec.count)

    def test_rejects_seed_invalid_counts_and_duplicate_revisions(self):
        for key, value in (
            ("seed", 7),
            ("concurrency", 5),
            ("repetitions", True),
            ("time_limit", float("nan")),
        ):
            raw = design()
            raw[key] = value
            with self.assertRaises(ContractError):
                CampaignSpec.accepted(raw)
        raw = design()
        raw["robots"]["robot-0"].append(raw["robots"]["robot-0"][0])
        with self.assertRaises(ContractError):
            CampaignSpec.accepted(raw)
        raw = design()
        raw["exclusions"] = [["robot-0", "world-0", {}]]
        with self.assertRaises(ContractError):
            CampaignSpec.accepted(raw)

    def test_45000_experiments_have_exact_distinct_identities(self):
        spec = CampaignSpec.accepted(design(10, 5, 3, 3, 100))
        self.assertEqual(spec.count, 45000)
        rows = list(experiments(spec, str(uuid.uuid4())))
        self.assertEqual(len(rows), spec.count)
        self.assertEqual(len({r.id for r in rows}), spec.count)

    def test_shared_worker_contract_is_immutable_and_rejects_invalid_domain(self):
        raw = dict(
            slot="0",
            run_id="a",
            domain_id=41,
            observer_domain=0,
            gazebo_partition="batch-a",
            process_token="owned-a",
            result_directory="/tmp/a",
            project_root="/tmp",
            session_root="/benchmark/a",
            scenario={"spawn": [1.0, 2.0, 0.1, 0.0]},
            single=False,
            spec={"overrides": {"weight": 2.0}},
            profile={},
            seed=1234,
            simulation_time_limit=1800.0,
            startup_timeout=240.0,
            robot={"id": "mogi"},
        )
        launch = WorkerLaunchSpec.from_configuration(raw)
        raw["scenario"]["spawn"][0] = 4.0
        self.assertEqual(launch.dictionary()["scenario"]["spawn"][0], 1.0)
        raw["domain_id"] = True
        with self.assertRaises(ValueError):
            WorkerLaunchSpec.from_configuration(raw)
        raw["domain_id"] = 41
        del raw["observer_domain"]
        with self.assertRaises(ValueError):
            WorkerLaunchSpec.from_configuration(raw)


if __name__ == "__main__":
    unittest.main()
