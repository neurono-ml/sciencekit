"""Tests for the sklearn parity matrix checker.

Run with ``python3 -m unittest discover -s scripts/tests -v``.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import parity_tooling  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "parity"

DAG = {
    "lanes": [{"id": "alpha", "merge_order": ["alpha", "beta"]}],
    "changes": [
        {
            "id": "alpha",
            "state": "shipped",
            "lane": "alpha",
            "completion": {
                "openspec": "true",
                "tests": ["cargo test -p sciencekit_alpha"],
            },
        },
        {
            "id": "beta",
            "state": "planned",
            "lane": "alpha",
            "completion": {
                "openspec": "true",
                "tests": ["cargo test -p sciencekit_alpha beta"],
            },
        },
    ]
}


def load_fixture() -> dict:
    return parity_tooling.load_matrix(FIXTURES / "parity.yaml")


def check(matrix: dict) -> list[str]:
    api = parity_tooling.load_api_listing(FIXTURES / "sklearn-api.txt")
    return parity_tooling.check_matrix(matrix, api, DAG, FIXTURES / "archive")


class MatrixCheckTests(unittest.TestCase):
    def test_fixture_is_consistent(self) -> None:
        self.assertEqual(check(load_fixture()), [])

    def test_unknown_status_is_rejected(self) -> None:
        matrix = load_fixture()
        matrix["entries"][1]["status"] = "maybe"
        errors = check(matrix)
        self.assertTrue(any("sklearn.alpha.B" in error and "maybe" in error for error in errors))

    def test_duplicate_entry_is_rejected(self) -> None:
        matrix = load_fixture()
        matrix["entries"].append(copy.deepcopy(matrix["entries"][0]))
        errors = check(matrix)
        self.assertTrue(any("duplicate" in error and "sklearn.alpha.A" in error for error in errors))

    def test_missing_baseline_class_is_rejected(self) -> None:
        matrix = load_fixture()
        del matrix["entries"][2]
        errors = check(matrix)
        self.assertTrue(any("sklearn.beta.C" in error for error in errors), errors)

    def test_entry_outside_the_baseline_is_rejected(self) -> None:
        matrix = load_fixture()
        matrix["entries"].append(
            {
                "api": "sklearn.gamma.D",
                "family": "gamma",
                "kind": "estimator",
                "status": "absent",
                "reason": "not part of the pinned release",
            }
        )
        errors = check(matrix)
        self.assertTrue(any("sklearn.gamma.D" in error for error in errors), errors)

    def test_missing_tier_is_rejected(self) -> None:
        matrix = load_fixture()
        del matrix["entries"][1]["tier"]
        errors = check(matrix)
        self.assertTrue(any("sklearn.alpha.B" in error and "tier" in error for error in errors))

    def test_dangling_planned_owner_is_rejected(self) -> None:
        matrix = load_fixture()
        matrix["entries"][1]["change"] = "ghost"
        errors = check(matrix)
        self.assertTrue(any("sklearn.alpha.B" in error and "ghost" in error for error in errors))

    def test_absent_entry_needs_a_reason(self) -> None:
        matrix = load_fixture()
        matrix["entries"][2]["reason"] = ""
        errors = check(matrix)
        self.assertTrue(any("sklearn.beta.C" in error and "reason" in error for error in errors))

    def test_absent_entry_must_not_name_a_change(self) -> None:
        matrix = load_fixture()
        matrix["entries"][2]["change"] = "beta"
        errors = check(matrix)
        self.assertTrue(any("sklearn.beta.C" in error and "change" in error for error in errors))

    def test_implemented_entry_needs_a_covering_test(self) -> None:
        matrix = load_fixture()
        matrix["entries"][0]["change"] = "beta"
        matrix["entries"][0]["crate"] = "sciencekit_other"
        matrix["crates"].append("sciencekit_other")
        errors = check(matrix)
        self.assertTrue(
            any("sklearn.alpha.A" in error and "coverage" in error for error in errors),
            errors,
        )

    def test_archived_owner_supplies_the_coverage_test(self) -> None:
        matrix = load_fixture()
        matrix["entries"][0]["change"] = "2020-01-01-alpha"
        self.assertEqual(check(matrix), [])


class CoveragePageTests(unittest.TestCase):
    def test_rendered_page_lists_statuses_and_families(self) -> None:
        page = parity_tooling.render_coverage_markdown(load_fixture())
        self.assertIn("| implemented | 1 |", page)
        self.assertIn("| alpha | 1 | 1 | 0 | 2 |", page)


class CoverageReportTests(unittest.TestCase):
    def test_report_counts_statuses_and_families(self) -> None:
        report = parity_tooling.coverage_report(load_fixture())
        self.assertEqual(report["statuses"]["implemented"], 1)
        self.assertEqual(report["statuses"]["planned"], 1)
        self.assertEqual(report["statuses"]["absent"], 1)
        self.assertEqual(report["families"]["alpha"]["planned"], 1)


if __name__ == "__main__":
    unittest.main()
