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

import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from benchmark_suite.contracts import HIDDEN_GROUPS
from benchmark_suite.gui import ProfileEditor, Window
from benchmark_suite.gui_support import TableModel
from benchmark_suite.library import Library


class GuiTests(unittest.TestCase):
    def test_live_inspection_uses_selected_robot_footprint(self):
        self.window.selected_attempt = "active"
        self.window.status = {
            "active": [{"attempt": {"id": "active"}, "experiment": {"robot_id": "mogi"}}]
        }
        self.window.inspect_selected()
        expected = next(
            row["footprint"] for row in self.library.catalog()["robots"] if row["id"] == "mogi"
        )
        self.assertEqual(self.window.live_map.vertices, expected)
        self.assertEqual(self.window.inspection_id, "active")

    def test_campaign_change_ignores_prior_report_and_export_operations(self):
        self.window.transport.submit = lambda *args: None
        self.window.select_campaign({"id": "old"})
        exports, reports = self.window.export_browser, self.window.report_browser
        exports.prepare()
        reports.generate()
        export_tag, report_tag = exports.preparation_tag, reports.generation_tag
        self.window.select_campaign({"id": "new"})
        exports.status.setText("Current export")
        reports.detail.setPlainText("Current report")
        self.window.statusBar().showMessage("Current campaign")
        exports.received(export_tag, {"export_id": "old-export"})
        reports.received(report_tag, {"report_id": "old-report"})
        exports.failed(export_tag, "Old export error")
        reports.failed(report_tag, "Old report error")
        self.window.failed(export_tag, "Old export error")
        self.window.failed(report_tag, "Old report error")
        self.assertEqual(exports.status.text(), "Current export")
        self.assertEqual(reports.detail.toPlainText(), "Current report")
        self.assertEqual(self.window.statusBar().currentMessage(), "Current campaign")

    def test_campaign_change_clears_report_and_export_actions(self):
        self.window.transport.submit = lambda *args: None
        self.window.select_campaign({"id": "old"})
        exports, reports = self.window.export_browser, self.window.report_browser
        old_export = {"id": "export", "campaign_id": "old", "state": "FINISHED", "path": "/tmp/old"}
        exports.table.model.replace([old_export])
        exports.selected = old_export
        reports.select_report({"id": "report", "campaign_id": "old"})
        old_request = reports.request_tag
        reports.artifact = {"path": "/tmp/old.pdf"}
        reports.model.replace([reports.artifact])
        self.window.select_campaign({"id": "new"})
        self.assertIsNone(exports.selected)
        self.assertEqual(exports.table.model.rowCount(), 0)
        self.assertIsNone(reports.report_id)
        self.assertIsNone(reports.artifact)
        self.assertEqual(reports.model.rowCount(), 0)
        reports.received(old_request, {"artifacts": [{"path": "/tmp/old.pdf"}]})
        self.assertEqual(reports.model.rowCount(), 0)
        exports.selected = old_export
        with patch.object(QtWidgets.QFileDialog, "getExistingDirectory") as destination:
            exports.copy()
            destination.assert_not_called()

    def test_inspection_reply_from_previous_worker_is_ignored(self):
        self.window.inspection_id = "new-worker"
        self.window.inspection_sequence = 0
        self.window.received(
            "inspection",
            {"attempt_id": "previous-worker", "sequence": 99, "path": [[10, 20]]},
        )
        self.assertEqual(self.window.inspection_sequence, 0)

    def test_refresh_during_a_request_is_reissued_and_failure_clears_page_links(self):
        table = self.window.results
        calls = []
        self.window.transport.submit = lambda *args: calls.append(args)
        table.table, table.busy = "attempts", False
        table.refresh()
        table.reset_page()
        table._received(table.tag, {"rows": [], "total": 0})
        self.assertEqual(len(calls), 2)
        self.assertTrue(table.busy)
        table.next_cursor = {"reserved_ns": 1000, "id": "old", "direction": "after"}
        table._failed(table.tag, "Disconnected")
        table.page(100)
        self.assertIsNone(table.next_cursor)
        self.assertEqual(len(calls), 2)

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.library = Library(Path(__file__).resolve().parents[2], native=False)

    def setUp(self):
        self.window = Window(Path("/tmp/benchmark-suite-no-engine.sock"))
        self.window.received("catalog", self.library.catalog())

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_builder_import_keeps_accepted_inline_configuration(self):
        accepted = self.library.resolve_campaign(
            {
                "name": "Imported",
                "robots": {"mogi": ["Default"]},
                "worlds": {"warehouse": ["Default"]},
                "repetitions": 100,
                "concurrency": 4,
            }
        ).dictionary()
        self.window.builder.apply_design(accepted)
        rebuilt = self.library.resolve_campaign(self.window.builder.design()).dictionary()
        self.assertEqual(rebuilt, accepted)
        self.window.fill_revisions("library-mogi", {"rows": [], "next_offset": None})
        self.assertEqual(
            self.library.resolve_campaign(self.window.builder.design()).dictionary(), accepted
        )
        self.assertFalse(self.window.workers.cards)

    def test_parameter_editor_hides_managed_groups_and_keeps_small_floats(self):
        profile = self.library.resolve_profile("mogi", {})
        editor = ProfileEditor(self.window.transport, self.library.catalog(), profile)
        try:
            for name in editor.fields:
                metadata = self.library.schema.metadata[name]
                self.assertFalse(metadata["managed"])
                self.assertNotIn(metadata["group"], HIDDEN_GROUPS)
            editor.fields["max_linear_speed_vmax"].setText("0.000000123456789")
            self.assertEqual(
                editor.configuration()["requested"]["max_linear_speed_vmax"], 0.000000123456789
            )
            editor.reset()
            self.assertEqual(editor.configuration()["requested"], {})
        finally:
            editor.close()

    def test_loaded_physical_settings_limits_and_spawn_pose_keep_precision(self):
        values = {
            "movement_speed": 0.123456789012345,
            "turning_speed": 1.23456789012345,
            "lidar_range": 10.1234567890123,
        }
        profile = self.library.resolve_profile("mogi", {"robot_overrides": values})
        editor = ProfileEditor(self.window.transport, self.library.catalog(), profile)
        try:
            self.assertEqual(editor.configuration()["robot_overrides"], values)
        finally:
            editor.close()
        builder = self.window.builder
        builder.limit.setValue(0.04)
        self.assertEqual(builder.design()["time_limit"], 0.04)
        pose = (1.2345678901234567, -2.9876543210123456, 0.12345678901234567)
        self.window.spawn_editor.map_pose(*pose)
        self.assertEqual(self.window.spawn_editor.pose(), pose)

    def test_import_invalidates_submission_before_file_response(self):
        builder, calls = self.window.builder, []
        self.window.transport.submit = lambda *args: calls.append(args)
        builder.validated = {"name": "Previous campaign"}
        builder.submit_button.setEnabled(True)
        with patch.object(
            QtWidgets.QFileDialog, "getOpenFileName", return_value=("/tmp/new.yaml", "")
        ):
            builder.import_campaign()
        self.assertIsNone(builder.validated)
        self.assertFalse(builder.submit_button.isEnabled())
        builder.submit_design(True)
        builder.submit_design(False)
        self.assertEqual([call[1] for call in calls], ["local.read"])

    def test_filter_change_clears_selection_and_rejects_old_details(self):
        self.window.transport.submit = lambda *args: None
        self.window.result_selection = "prior-attempt"
        self.window.result_row = {"id": "prior-attempt", "robot_id": "mogi"}
        self.window.result_detail.setPlainText("Previous details")
        self.window.filter_results()
        self.window.received("result-detail", {"id": "prior-attempt", "metrics": {"coverage": 50}})
        self.assertIsNone(self.window.result_selection)
        self.assertFalse(self.window.result_row)
        self.assertNotIn("Previous details", self.window.result_detail.toPlainText())
        self.assertNotIn("50", self.window.result_detail.toPlainText())

    def test_unresolved_outcome_is_counted_once_in_campaign_progress(self):
        self.window.campaign_id = "campaign"
        self.window.received(
            "campaign-status",
            {
                "campaign": {"id": "campaign", "name": "Study", "count": 3, "state": "RUNNING"},
                "counters": {
                    "outcomes": {"completed": 1, "timing_rejection": 1},
                    "states": {"DONE": 1, "UNRESOLVED": 1, "PENDING": 1},
                    "attempts": {
                        "executed": 3,
                        "timing_rejections": 2,
                        "sequential_replacements": 1,
                        "discards": 0,
                    },
                },
            },
        )
        self.assertEqual(self.window.progress_bar.value(), 667)

    def test_large_table_has_only_one_page_and_fast_model_access(self):
        model = TableModel([("id", "Attempt"), ("phase", "Phase")])
        started = time.perf_counter()
        for page in range(450):
            model.replace([{"id": page * 100 + i, "phase": "FINISHED"} for i in range(100)])
            self.assertEqual(model.rowCount(), 100)
            self.assertEqual(model.data(model.index(0, 1)), "Finished")
        elapsed = time.perf_counter() - started
        print(f"450 table pages: {elapsed:.3f}s; final rows {model.rowCount()}")
        self.assertLess(elapsed, 2.0)

    def test_world_pose_round_trip_uses_radians(self):
        self.window.spawn_editor.map_pose(1.2345, -2.3456, 0.7)
        self.assertAlmostEqual(self.window.spawn_editor.spawn_map.pose[0], 1.2345)
        self.assertAlmostEqual(self.window.spawn_editor.spawn_map.pose[1], -2.3456)
        self.assertAlmostEqual(self.window.spawn_editor.spawn_map.pose[2], 0.7, places=5)

    def test_reconnection_reloads_catalog_and_preserves_browsed_page(self):
        calls = []
        self.window.transport.submit = lambda *args: calls.append(args)
        self.window.catalog = {}
        self.window.catalog_pending = False
        self.window.campaigns.offset = 100
        self.window.campaigns.model.replace([{"id": "later-page"}])
        self.window.received(
            "status",
            {
                "backend": "synthetic",
                "active": [],
                "workspace": "/tmp",
                "campaigns": {"total": 200, "rows": [{"id": "first-page"}]},
            },
        )
        self.assertTrue(any(args[:2] == ("catalog", "catalog.get") for args in calls))
        self.assertEqual(self.window.campaigns.model.rows[0]["id"], "later-page")
        self.window.results.offset = 100
        self.window.select_campaign({"id": "another"})
        self.assertEqual(self.window.results.offset, 0)

    def test_inflight_page_response_cannot_overwrite_changed_selection(self):
        calls = []
        self.window.transport.submit = lambda *args: calls.append(args)
        table = self.window.results
        table.busy = False
        table.refresh()
        table.offset = 100
        table.refresh()
        table._received(table.tag, {"rows": [{"id": "old-page"}], "total": 300})
        self.assertFalse(table.model.rows)
        self.assertEqual(calls[-1][2]["offset"], 100)
        table._received(table.tag, {"rows": [{"id": "new-page"}], "total": 300})
        self.assertEqual(table.model.rows[0]["id"], "new-page")

    def test_guided_builder_requires_world_and_current_validation(self):
        builder = self.window.builder
        calls = []
        self.window.transport.submit = lambda *args: calls.append(args)
        builder.next_step()
        self.assertEqual(builder.step, 1)
        builder.next_step()
        self.assertEqual(builder.step, 1)
        self.assertIn("Select at least one world", builder.error.text())
        builder.world_tree.topLevelItem(2).setCheckState(0, QtCore.Qt.CheckState.Checked)
        builder.show_step(3)
        builder.submit_design(False)
        old = builder.validation_tag
        accepted = self.library.resolve_campaign(builder.design()).dictionary()
        result = {
            "specification": accepted,
            "count": 1,
            "expected_minimum_attempts": 1,
            "timing_retry": "One final sequential replacement",
        }
        builder.repetitions.setValue(10)
        builder.received(old, result)
        self.assertIsNone(builder.validated)
        builder.submit_design(True)
        self.assertFalse(any(call[1] == "campaign.submit" for call in calls))
        builder.submit_design(False)
        current = builder.validation_tag
        builder.received(current, result)
        builder.submit_design(True)
        builder.submit_design(True)
        self.assertEqual(sum(call[1] == "campaign.submit" for call in calls), 1)

    def test_builder_import_response_restores_accepted_configuration(self):
        builder = self.window.builder
        self.window.transport.submit = lambda *args: None
        accepted = self.library.resolve_campaign(
            {
                "name": "Imported study",
                "robots": {"mogi": ["Default"]},
                "worlds": {"corridor": ["Default"]},
                "concurrency": 3,
                "repetitions": 10,
            }
        ).dictionary()
        builder.validation_tag = "campaign-import-example"
        builder.received(
            builder.validation_tag,
            {
                "specification": accepted,
                "count": 10,
                "expected_minimum_attempts": 10,
                "timing_retry": "One replacement",
            },
        )
        self.assertEqual(self.library.resolve_campaign(builder.design()).dictionary(), accepted)
        self.assertTrue(builder.submit_button.isEnabled())

    def test_readable_exclusions_follow_selected_combinations(self):
        builder = self.window.builder
        builder.robot_selector.add_profile("tb3_burger", "Default", "Default")
        builder.world_tree.topLevelItem(0).setCheckState(0, QtCore.Qt.CheckState.Checked)
        builder.world_tree.topLevelItem(2).setCheckState(0, QtCore.Qt.CheckState.Checked)
        builder.show_step(3)
        self.assertEqual(builder.compatibility.rowCount(), 4)
        item = builder.compatibility.item(0, 0)
        identity = item.data(QtCore.Qt.ItemDataRole.UserRole)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        accepted = self.library.resolve_campaign(builder.design())
        self.assertEqual(accepted.count, 3)
        builder.world_tree.topLevelItem(0).setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        self.assertNotIn(list(identity), builder.design()["exclusions"])

    def test_page_navigation_never_submits_or_starts_simulations(self):
        calls = []
        self.window.transport.submit = lambda *args: calls.append(args)
        for key in ("monitor", "profiles", "worlds", "results", "builder", "campaigns"):
            self.window.go(key)
            self.assertEqual(self.window.current_page(), key)
        self.assertEqual(self.window.navigation.count(), 5)
        self.assertFalse(any(call[1] in {"campaign.start", "campaign.submit"} for call in calls))

    def test_refresh_preserves_selected_record_and_column_changes_reset(self):
        table = self.window.results
        self.window.transport.submit = lambda *args: None
        table.model.replace([{"id": "a"}, {"id": "b"}])
        table.view.setCurrentIndex(table.model.index(1, 0))
        table.busy = False
        table.refresh()
        table._received(table.tag, {"rows": [{"id": "b"}, {"id": "a"}], "total": 2})
        selected = table.model.rows[table.view.currentIndex().row()]
        self.assertEqual(selected["id"], "b")
        table.model.set_columns([("id", "Identity")])
        self.assertEqual(table.model.columnCount(), 1)
        self.assertEqual(table.model.rowCount(), 0)

    def test_worker_cards_reuse_widgets_and_retain_selected_metrics(self):
        self.window.transport.submit = lambda *args: None
        active = [
            {
                "attempt": {
                    "id": f"a-{i}",
                    "number": 1,
                    "phase": "RUNNING",
                    "directory": "/tmp/run",
                },
                "experiment": {
                    "id": f"e-{i}",
                    "robot_id": "mogi",
                    "world_id": "warehouse",
                    "profile_name": "Default",
                    "spawn_name": "Default",
                    "repetition": i,
                },
                "metrics": {"coverage_percent": 50.0, "simulation_elapsed": 10.0},
            }
            for i in range(4)
        ]
        status = {
            "active": active,
            "backend": "gazebo",
            "workspace": "/tmp",
            "campaigns": {"total": 1},
        }
        self.window.received("status", status)
        self.window.choose_worker("a-2")
        card = self.window.workers.cards["a-2"]
        self.window.received("status", status)
        self.assertIs(self.window.workers.cards["a-2"], card)
        self.assertEqual(self.window.workers.selected, "a-2")
        self.window.received("status", {**status, "active": []})
        self.assertFalse(self.window.stop_run_button.isEnabled())
        self.assertEqual(self.window.run_metrics.values["coverage"].text(), "50.0%")
        self.assertFalse(self.window.workers.cards)


if __name__ == "__main__":
    unittest.main()
