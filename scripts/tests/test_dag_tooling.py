"""Tests for the implementation DAG tooling.

Run with ``python3 -m unittest discover -s scripts/tests -v``.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dag_tooling  # noqa: E402

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "dag_fixture.yaml"


def load_fixture() -> dict:
    return dag_tooling.load_dag(FIXTURE)


class ConsistencyCheckTests(unittest.TestCase):
    def test_fixture_is_consistent(self) -> None:
        errors = dag_tooling.check_dag(load_fixture(), design_path=None)
        self.assertEqual(errors, [])

    def test_duplicate_change_id_is_reported(self) -> None:
        dag = load_fixture()
        dag["changes"].append(copy.deepcopy(dag["changes"][0]))
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("duplicate" in error for error in errors), errors)

    def test_unresolved_dependency_is_reported(self) -> None:
        dag = load_fixture()
        dag["changes"][1]["depends_on"] = ["missing"]
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("missing" in error for error in errors), errors)

    def test_unknown_lane_is_reported(self) -> None:
        dag = load_fixture()
        dag["changes"][1]["lane"] = "nowhere"
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("nowhere" in error for error in errors), errors)

    def test_missing_completion_block_is_reported(self) -> None:
        dag = load_fixture()
        del dag["changes"][1]["completion"]
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("completion" in error for error in errors), errors)

    def test_dependency_cycle_is_reported(self) -> None:
        dag = load_fixture()
        dag["changes"][0]["depends_on"] = ["beta"]
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("cycle" in error for error in errors), errors)

    def test_same_lane_dependency_order_violation_is_reported(self) -> None:
        dag = load_fixture()
        dag["lanes"][0]["merge_order"] = ["beta", "alpha"]
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("merge_order" in error for error in errors), errors)

    def test_wave_member_must_exist(self) -> None:
        dag = load_fixture()
        dag["execution_waves"][0]["changes"].append("ghost")
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("ghost" in error for error in errors), errors)

    def test_active_change_belongs_to_exactly_one_wave(self) -> None:
        dag = load_fixture()
        dag["execution_waves"][1]["changes"].append("beta")
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("wave" in error for error in errors), errors)

    def test_superseded_change_needs_a_resolution(self) -> None:
        dag = load_fixture()
        dag["changes"][3]["state"] = "superseded"
        errors = dag_tooling.check_dag(dag, design_path=None)
        self.assertTrue(any("superseded_by" in error for error in errors), errors)


class ReadinessTests(unittest.TestCase):
    def test_states_follow_the_merged_set(self) -> None:
        dag = load_fixture()
        states = dag_tooling.compute_states(dag, merged={"alpha"})
        self.assertEqual(states["alpha"], "done")
        self.assertEqual(states["beta"], "ready")
        self.assertEqual(states["gamma"], "ready")
        self.assertEqual(states["delta"], "blocked")

    def test_shipped_and_superseded_are_done(self) -> None:
        dag = load_fixture()
        dag["changes"][2]["state"] = "superseded"
        dag["changes"][2]["superseded_by"] = ["alpha"]
        states = dag_tooling.compute_states(dag, merged=set())
        self.assertEqual(states["alpha"], "done")
        self.assertEqual(states["gamma"], "done")

    def test_deferred_is_never_ready(self) -> None:
        dag = load_fixture()
        dag["changes"][1]["state"] = "deferred"
        dag["changes"][1]["deferred"] = True
        states = dag_tooling.compute_states(dag, merged={"alpha"})
        self.assertEqual(states["beta"], "deferred")

    def test_next_per_lane_prefers_merge_order(self) -> None:
        dag = load_fixture()
        next_changes = dag_tooling.next_per_lane(dag, merged={"alpha"})
        self.assertEqual(next_changes["lane-one"], "beta")
        self.assertEqual(next_changes["lane-two"], "gamma")

    def test_occupied_lane_has_no_next(self) -> None:
        dag = load_fixture()
        dag["changes"][1]["state"] = "in-progress"
        next_changes = dag_tooling.next_per_lane(dag, merged={"alpha"})
        self.assertIsNone(next_changes["lane-one"])
        self.assertEqual(next_changes["lane-two"], "gamma")


class CompletionTests(unittest.TestCase):
    def test_dry_run_collects_openspec_tests_and_gates(self) -> None:
        dag = load_fixture()
        plan = dag_tooling.completion_plan(dag, "beta", include_global=True)
        self.assertEqual(plan.openspec_command, "true")
        self.assertEqual(list(plan.test_commands), ["true"])
        self.assertEqual(list(plan.global_gate_commands), ["true"])

    def test_unknown_change_is_rejected(self) -> None:
        dag = load_fixture()
        with self.assertRaises(KeyError):
            dag_tooling.completion_plan(dag, "ghost")


if __name__ == "__main__":
    unittest.main()
