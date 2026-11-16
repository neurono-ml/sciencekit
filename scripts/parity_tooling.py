"""Shared logic for the sklearn parity matrix checker.

The matrix lives in ``openspec/changes/sklearn-parity-matrix/parity.yaml``; its
inventory is the pinned API listing and its owners are the implementation DAG
changes (or an archived change). Covered by ``scripts/tests/test_parity_tooling.py``.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "openspec" / "changes" / "sklearn-parity-matrix" / "parity.yaml"
DAG_PATH = ROOT / "openspec" / "changes" / "implementation-dag" / "dag.yaml"
ARCHIVE_PATH = ROOT / "openspec" / "changes" / "archive"

STATUSES = {"implemented", "planned", "absent"}
TIERS = {"streaming", "mappable", "chunked", "full-memory"}
TEST_COMMAND = re.compile(r"cargo test[^`\n]*")


def load_matrix(path: str | pathlib.Path = MATRIX_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dag(path: str | pathlib.Path = DAG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_api_listing(path: str | pathlib.Path) -> list[str]:
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def resolve_owner_tests(
    owner: str,
    dag: dict[str, Any],
    archive_dir: pathlib.Path,
) -> list[str] | None:
    """Return the owner's completion test commands, or ``None`` when it does not exist."""

    for change in dag.get("changes", []):
        if change.get("id") == owner:
            completion = change.get("completion") or {}
            return [str(command) for command in completion.get("tests") or []]
    tasks = archive_dir / owner / "tasks.md"
    if tasks.exists():
        return TEST_COMMAND.findall(tasks.read_text(encoding="utf-8"))
    return None


def _covers(tests: list[str], crate: str) -> bool:
    return any(crate in command or "--workspace" in command for command in tests)


def check_matrix(
    matrix: dict[str, Any],
    api_lines: list[str],
    dag: dict[str, Any],
    archive_dir: pathlib.Path = ARCHIVE_PATH,
) -> list[str]:
    """Return every consistency error; an empty list means the matrix is valid."""

    errors: list[str] = []
    crates = set(matrix.get("crates") or [])
    lanes = {lane.get("id") for lane in dag.get("lanes") or []}
    known_changes = {change.get("id"): change for change in dag.get("changes") or []}
    entries = matrix.get("entries") or []

    seen: dict[str, int] = {}
    for entry in entries:
        api = entry.get("api", "<missing>")
        seen[api] = seen.get(api, 0) + 1
        if seen[api] == 2:
            errors.append(f"{api}: duplicate matrix entry")

    statuses = [entry.get("status") for entry in entries]
    for entry, status in zip(entries, statuses):
        api = entry.get("api", "<missing>")
        if status not in STATUSES:
            errors.append(f"{api}: status {status!r} is not one of {sorted(STATUSES)}")

    baseline = set(api_lines)
    estimator_entries = [
        entry for entry in entries if entry.get("kind") == "estimator"
    ]
    listed = {entry.get("api") for entry in estimator_entries}
    for api in sorted(baseline - listed):
        errors.append(f"{api}: baseline class has no matrix entry")
    for api in sorted(listed - baseline):
        errors.append(f"{api}: matrix entry is not part of the pinned baseline")

    for entry in entries:
        api = entry.get("api", "<missing>")
        status = entry.get("status")
        crate = entry.get("crate")
        if status in {"implemented", "planned"}:
            if entry.get("tier") not in TIERS:
                errors.append(
                    f"{api}: tier {entry.get('tier')!r} is missing or outside {sorted(TIERS)}"
                )
            if crate not in crates:
                errors.append(f"{api}: crate {crate!r} is not declared in crates")
        if status == "planned":
            owner = entry.get("change")
            change = known_changes.get(owner)
            if change is None:
                errors.append(f"{api}: planned owner {owner!r} is absent from the DAG")
            elif change.get("lane") not in lanes:
                errors.append(
                    f"{api}: owner {owner!r} is not reachable from a DAG lane"
                )
        elif status == "implemented":
            owner = entry.get("change")
            if not owner:
                errors.append(f"{api}: implemented entry needs a change owner")
                continue
            tests = resolve_owner_tests(owner, dag, pathlib.Path(archive_dir))
            if tests is None:
                errors.append(f"{api}: owner {owner!r} is absent from the DAG and archive")
            elif not _covers(tests, crate or ""):
                errors.append(
                    f"{api}: coverage test is missing for crate {crate!r} "
                    f"in owner {owner!r}"
                )
        elif status == "absent":
            if not (entry.get("reason") or "").strip():
                errors.append(f"{api}: absent entry needs a non-empty reason")
            if entry.get("change"):
                errors.append(f"{api}: absent entry must not name a change")

    return errors


def coverage_report(matrix: dict[str, Any]) -> dict[str, dict[str, int]]:
    statuses: dict[str, int] = {}
    families: dict[str, dict[str, int]] = {}
    for entry in matrix.get("entries") or []:
        status = entry.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        family = entry.get("family", "unknown")
        families.setdefault(family, {})
        families[family][status] = families[family].get(status, 0) + 1
    return {"statuses": statuses, "families": families}


def main(path: str | pathlib.Path = MATRIX_PATH) -> int:
    matrix = load_matrix(path)
    reference = pathlib.Path(path).parent / str(matrix.get("reference", ""))
    api = load_api_listing(reference)
    errors = check_matrix(matrix, api, load_dag(), ARCHIVE_PATH)
    if errors:
        print(f"{len(errors)} parity matrix error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    report = coverage_report(matrix)
    print(
        "parity matrix is consistent: "
        + ", ".join(f"{status}={count}" for status, count in sorted(report["statuses"].items()))
    )
    return 0


def render_coverage_markdown(matrix: dict[str, Any]) -> str:
    """Render the coverage page from the matrix (the source of truth)."""

    report = coverage_report(matrix)
    statuses = report["statuses"]
    families = report["families"]
    lines = [
        "# scikit-learn parity coverage",
        "",
        "Generated from `openspec/changes/sklearn-parity-matrix/parity.yaml`; do not edit",
        "this page by hand. Regenerate it with `python3 scripts/render-parity-coverage.py`.",
        "",
        f"Pinned baseline: scikit-learn {matrix.get('baseline', 'unknown')}.",
        "",
        "## Status",
        "",
        "| Status | Entries |",
        "|---|---:|",
    ]
    for status in ("implemented", "planned", "absent"):
        lines.append(f"| {status} | {statuses.get(status, 0)} |")
    lines.append(f"| **total** | **{sum(statuses.values())}** |")
    lines.extend(
        [
            "",
            "## Families",
            "",
            "| Family | implemented | planned | absent | total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family in sorted(families):
        counts = families[family]
        total = sum(counts.values())
        lines.append(
            f"| {family} | {counts.get('implemented', 0)} | {counts.get('planned', 0)} "
            f"| {counts.get('absent', 0)} | {total} |"
        )
    lines.append("")
    return "\n".join(lines)


COVERAGE_PAGE = ROOT / "docs" / "src" / "parity-coverage.md"


def main_render(
    matrix_path: str | pathlib.Path = MATRIX_PATH,
    output_path: str | pathlib.Path = COVERAGE_PAGE,
    check: bool = False,
) -> int:
    matrix = load_matrix(matrix_path)
    rendered = render_coverage_markdown(matrix)
    output = pathlib.Path(output_path)
    if check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current != rendered:
            print(f"{output} is stale; run scripts/render-parity-coverage.py")
            return 1
        print(f"{output} is up to date")
        return 0
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {output}")
    return 0
