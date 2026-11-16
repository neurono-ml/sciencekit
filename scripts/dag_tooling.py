"""Shared logic for the implementation DAG tooling.

The DAG is ``openspec/changes/implementation-dag/dag.yaml``; it is the single source
of truth for change states, dependencies, lanes and completion checks. These helpers
are imported by ``check-dag.py``, ``dag-readiness.py`` and ``check-completion.py`` and
covered by ``scripts/tests/test_dag_tooling.py``.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import subprocess
from typing import Any, Iterable, Sequence

import yaml

DAG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "openspec"
    / "changes"
    / "implementation-dag"
    / "dag.yaml"
)
DESIGN_PATH = DAG_PATH.with_name("design.md")

VALID_STATES = {"shipped", "closing", "in-progress", "planned", "deferred", "superseded"}

# States that mean the change is finished for scheduling purposes.
DONE_STATES = {"shipped", "superseded"}
# States that keep a lane busy even though the change is not finished.
LANE_OCCUPYING_STATES = {"in-progress", "closing"}


def load_dag(path: str | pathlib.Path = DAG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return data


def changes_by_id(dag: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {change["id"]: change for change in dag.get("changes", [])}


def lane_ids(dag: dict[str, Any]) -> list[str]:
    return [lane["id"] for lane in dag.get("lanes", [])]


def active_changes(dag: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        change
        for change in dag.get("changes", [])
        if change.get("state") not in DONE_STATES and change.get("state") != "deferred"
    ]


def _dependency_cycle(dag: dict[str, Any]) -> list[str] | None:
    known = changes_by_id(dag)
    white, grey, black = 0, 1, 2
    colour = {change_id: white for change_id in known}
    stack: list[str] = []

    def visit(change_id: str) -> list[str] | None:
        colour[change_id] = grey
        stack.append(change_id)
        for dependency in known[change_id].get("depends_on") or []:
            if dependency not in known:
                continue
            if colour[dependency] == grey:
                start = stack.index(dependency)
                return stack[start:] + [dependency]
            if colour[dependency] == white:
                found = visit(dependency)
                if found:
                    return found
        stack.pop()
        colour[change_id] = black
        return None

    for change_id in known:
        if colour[change_id] == white:
            found = visit(change_id)
            if found:
                return found
    return None


def _design_lane_rows(design_path: pathlib.Path) -> set[str]:
    text = design_path.read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*([a-z0-9-]+)\s*\|", text, flags=re.MULTILINE))


def check_dag(
    dag: dict[str, Any],
    design_path: pathlib.Path | None = DESIGN_PATH,
) -> list[str]:
    """Return every consistency error in the graph; an empty list means valid."""

    errors: list[str] = []
    changes = dag.get("changes")
    if not isinstance(changes, list):
        return ["changes: missing or not a list"]

    known = changes_by_id(dag)
    identifier_counts: dict[str, int] = {}
    for change in changes:
        identifier_counts[change.get("id", "<missing>")] = (
            identifier_counts.get(change.get("id", "<missing>"), 0) + 1
        )
    for identifier, count in identifier_counts.items():
        if count > 1:
            errors.append(f"{identifier}: duplicate change id ({count} times)")

    lanes = lane_ids(dag)
    if len(lanes) != len(set(lanes)):
        errors.append("lanes: duplicate lane id")
    known_lanes = set(lanes)

    for change in changes:
        identifier = change.get("id", "<missing>")
        if not change.get("lane"):
            errors.append(f"{identifier}: missing lane")
        elif change["lane"] not in known_lanes:
            errors.append(f"{identifier}: unknown lane {change['lane']!r}")
        if change.get("state") not in VALID_STATES:
            errors.append(f"{identifier}: unknown or missing state {change.get('state')!r}")
        completion = change.get("completion")
        if not isinstance(completion, dict):
            errors.append(f"{identifier}: missing completion block")
        else:
            if not completion.get("openspec"):
                errors.append(f"{identifier}: completion.openspec is empty")
            tests = completion.get("tests")
            if tests is None:
                errors.append(f"{identifier}: completion.tests is missing")
            elif not tests and not change.get("docs_only"):
                errors.append(
                    f"{identifier}: empty completion.tests needs docs_only: true"
                )
        for dependency in change.get("depends_on") or []:
            if dependency not in known:
                errors.append(f"{identifier}: dependency {dependency!r} is unresolved")
            if dependency == identifier:
                errors.append(f"{identifier}: depends on itself")
        for dependency in change.get("soft_depends_on") or []:
            if dependency not in known:
                errors.append(f"{identifier}: soft dependency {dependency!r} is unresolved")
        if change.get("state") == "superseded":
            resolutions = change.get("superseded_by") or []
            if not resolutions:
                errors.append(f"{identifier}: superseded change needs superseded_by")
            for resolution in resolutions:
                if resolution not in known:
                    errors.append(
                        f"{identifier}: superseded_by {resolution!r} is unresolved"
                    )

    cycle = _dependency_cycle(dag)
    if cycle:
        errors.append("dependency cycle: " + " -> ".join(cycle))

    wave_membership: dict[str, list[str]] = {}
    for wave in dag.get("execution_waves") or []:
        wave_id = wave.get("id", "<missing>")
        members = wave.get("changes") or []
        if not members:
            errors.append(f"wave {wave_id}: no changes")
        for member in members:
            if member not in known:
                errors.append(f"wave {wave_id}: unknown change {member!r}")
                continue
            wave_membership.setdefault(member, []).append(wave_id)
    for change in changes:
        identifier = change.get("id")
        if change.get("state") in DONE_STATES or change.get("state") == "deferred":
            continue
        waves = wave_membership.get(identifier, [])
        if len(waves) == 0:
            errors.append(f"{identifier}: active change belongs to no execution wave")
        elif len(waves) > 1:
            errors.append(
                f"{identifier}: active change belongs to several waves ({', '.join(waves)})"
            )

    wave_index: dict[str, int] = {}
    for index, wave in enumerate(dag.get("execution_waves") or []):
        for member in wave.get("changes") or []:
            wave_index[member] = index
    for change in changes:
        identifier = change.get("id")
        if change.get("state") in DONE_STATES | {"superseded", "deferred"}:
            continue
        for dependency in change.get("depends_on") or []:
            if dependency not in wave_index or identifier not in wave_index:
                continue
            if wave_index[dependency] > wave_index[identifier]:
                errors.append(
                    f"{identifier}: depends on {dependency!r} scheduled in a later wave"
                )

    members_of_lane: dict[str, set[str]] = {lane.get("id"): set() for lane in dag.get("lanes") or []}
    positions: dict[str, dict[str, int]] = {}
    for lane in dag.get("lanes") or []:
        lane_id = lane.get("id")
        order = lane.get("merge_order") or []
        positions[lane_id] = {}
        for index, member in enumerate(order):
            if member not in known:
                errors.append(f"lane {lane_id}: merge_order member {member!r} is unknown")
            if member in positions[lane_id]:
                errors.append(f"lane {lane_id}: duplicate merge_order member {member!r}")
            positions[lane_id][member] = index
            members_of_lane[lane_id].add(member)
            owning = known.get(member)
            if owning is not None and owning.get("lane") != lane_id:
                errors.append(
                    f"lane {lane_id}: {member!r} declares lane {owning.get('lane')!r}"
                )
        for member in order:
            owning = known.get(member)
            if owning is None:
                continue
            for dependency in owning.get("depends_on") or []:
                if dependency not in positions[lane_id]:
                    continue
                if positions[lane_id][dependency] > positions[lane_id][member]:
                    errors.append(
                        f"merge_order: {member!r} depends on same-lane {dependency!r} "
                        "that comes later"
                    )

    for change in changes:
        identifier = change.get("id")
        if change.get("state") in DONE_STATES | {"superseded", "deferred"}:
            continue
        lane_id = change.get("lane")
        if lane_id in members_of_lane and identifier not in members_of_lane[lane_id]:
            errors.append(f"{identifier}: active change is absent from {lane_id} merge_order")

    if design_path is not None and design_path.exists():
        documented = _design_lane_rows(design_path)
        for lane in lanes:
            if lane not in documented:
                errors.append(f"design.md: lane {lane!r} is not documented")

    return errors


def compute_layers(dag: dict[str, Any]) -> dict[int, list[str]]:
    """Longest-path layer per change over ``depends_on`` (superseded nodes excluded)."""

    known = changes_by_id(dag)
    depth: dict[str, int] = {}

    def resolve(identifier: str) -> int:
        if identifier in depth:
            return depth[identifier]
        dependencies = [
            dependency
            for dependency in known[identifier].get("depends_on") or []
            if dependency in known
            and known[dependency].get("state") != "superseded"
        ]
        depth[identifier] = 0 if not dependencies else 1 + max(resolve(dep) for dep in dependencies)
        return depth[identifier]

    for identifier, change in known.items():
        if change.get("state") != "superseded":
            resolve(identifier)
    layers: dict[int, list[str]] = {}
    for identifier, level in depth.items():
        layers.setdefault(level, []).append(identifier)
    return {level: sorted(members) for level, members in sorted(layers.items())}


def render_layers(dag: dict[str, Any]) -> str:
    lines = ["| Layer | Changes |", "|---|---|"]
    for level, members in compute_layers(dag).items():
        lines.append(f"| L{level} | {', '.join(members)} |")
    return "\n".join(lines)


def render_lanes(dag: dict[str, Any]) -> str:
    lines = ["| Lane | Merge order |", "|---|---|"]
    for lane in dag.get("lanes") or []:
        order = " → ".join(lane.get("merge_order") or [])
        note = " (deferred)" if lane.get("deferred") else ""
        lines.append(f"| {lane.get('id')} | {order}{note} |")
    return "\n".join(lines)


LAYERS_MARKER = ("<!-- dag-layers:start -->", "<!-- dag-layers:end -->")
LANES_MARKER = ("<!-- dag-lanes:start -->", "<!-- dag-lanes:end -->")


def replace_block(text: str, markers: tuple[str, str], body: str) -> str:
    start, end = markers
    if start not in text or end not in text:
        raise ValueError(f"missing marker {start} / {end}")
    prefix, rest = text.split(start, 1)
    _, suffix = rest.split(end, 1)
    return f"{prefix}{start}\n{body}\n{end}{suffix}"


def design_sync_errors(dag: dict[str, Any], design_path: pathlib.Path) -> list[str]:
    text = design_path.read_text(encoding="utf-8")
    errors: list[str] = []
    for markers, expected, label in (
        (LAYERS_MARKER, render_layers(dag), "layers"),
        (LANES_MARKER, render_lanes(dag), "lanes"),
    ):
        start, end = markers
        if start not in text or end not in text:
            errors.append(f"design.md: missing the generated {label} block")
            continue
        current = text.split(start, 1)[1].split(end, 1)[0].strip()
        if current != expected:
            errors.append(f"design.md: the {label} block is stale; rerender it")
    return errors


def compute_states(
    dag: dict[str, Any],
    merged: Iterable[str],
) -> dict[str, str]:
    """Return ``done``/``ready``/``blocked``/``deferred`` for every change."""

    merged_set = set(merged)
    known = changes_by_id(dag)
    states: dict[str, str] = {}

    def resolve(identifier: str) -> str:
        if identifier in states:
            return states[identifier]
        change = known[identifier]
        if identifier in merged_set or change.get("state") in DONE_STATES:
            states[identifier] = "done"
        elif change.get("deferred") or change.get("state") == "deferred":
            states[identifier] = "deferred"
        elif all(
            resolve(dependency) in {"done", "deferred"}
            for dependency in change.get("depends_on") or []
            if dependency in known
        ):
            states[identifier] = "ready"
        else:
            states[identifier] = "blocked"
        return states[identifier]

    for identifier in known:
        resolve(identifier)
    return states


def next_per_lane(dag: dict[str, Any], merged: Iterable[str]) -> dict[str, str | None]:
    """Return the next runnable change per lane, honouring merge order and occupancy."""

    states = compute_states(dag, merged)
    known = changes_by_id(dag)
    result: dict[str, str | None] = {}
    for lane in dag.get("lanes") or []:
        lane_id = lane["id"]
        order = lane.get("merge_order") or []
        occupied = any(
            known[member].get("state") in LANE_OCCUPYING_STATES
            or states.get(member) == "in-progress"
            for member in order
            if member in known
        )
        if occupied:
            result[lane_id] = None
            continue
        chosen: str | None = None
        for member in order:
            if member not in known:
                continue
            state = states.get(member)
            if state in {"done", "deferred"}:
                continue
            if state == "ready":
                chosen = member
            break
        result[lane_id] = chosen
    return result


@dataclasses.dataclass(frozen=True)
class CompletionPlan:
    change_id: str
    openspec_command: str
    test_commands: tuple[str, ...]
    extra_commands: tuple[str, ...]
    global_gate_commands: tuple[str, ...]
    docs_command: str | None


def completion_plan(
    dag: dict[str, Any],
    change_id: str,
    include_global: bool = True,
) -> CompletionPlan:
    change = changes_by_id(dag)[change_id]
    completion = change.get("completion") or {}
    extra = completion.get("extra") or []
    docs_command = next(
        (command for command in extra if "mdbook build" in command), None
    )
    if include_global:
        policy = dag.get("policy") or {}
        merge_requires = policy.get("merge_requires")
        if isinstance(merge_requires, dict):
            gates = tuple(merge_requires.get("global_gates") or [])
        else:
            gates = tuple(policy.get("global_gates") or [])
    else:
        gates = ()
    return CompletionPlan(
        change_id=change_id,
        openspec_command=str(completion.get("openspec", "")),
        test_commands=tuple(str(command) for command in completion.get("tests") or []),
        extra_commands=tuple(str(command) for command in extra),
        global_gate_commands=gates,
        docs_command=docs_command,
    )


def unmerged_dependencies(
    dag: dict[str, Any], change_id: str, merged: Iterable[str] = ()
) -> list[str]:
    """Return the hard dependencies that are not done yet."""

    states = compute_states(dag, merged)
    change = changes_by_id(dag)[change_id]
    return [
        dependency
        for dependency in change.get("depends_on") or []
        if states.get(dependency) not in {"done", "deferred"}
    ]


def run_command(command: str) -> tuple[int, str]:
    """Run one check command from the repository root and return its exit status."""

    root = pathlib.Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        command,
        shell=True,
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output.strip()


def report(name: str, status: int, output: str, verbose: bool = True) -> None:
    marker = "PASS" if status == 0 else "FAIL"
    print(f"[{marker}] {name}")
    if verbose and status != 0 and output:
        tail = "\n".join(output.splitlines()[-30:])
        print(tail)


def main_check(
    dag_path: str | pathlib.Path = DAG_PATH,
    design_path: pathlib.Path | None = DESIGN_PATH,
) -> int:
    dag = load_dag(dag_path)
    errors = check_dag(dag)
    if design_path is not None and design_path.exists():
        errors.extend(design_sync_errors(dag, design_path))
    if errors:
        print(f"{len(errors)} consistency error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        f"dag is consistent: {len(dag.get('changes', []))} changes, "
        f"{len(dag.get('lanes', []))} lanes, "
        f"{len(dag.get('execution_waves', []))} waves"
    )
    return 0


def main_readiness(
    dag_path: str | pathlib.Path = DAG_PATH,
    merged: Sequence[str] = (),
    as_json: bool = False,
) -> int:
    dag = load_dag(dag_path)
    errors = check_dag(dag)
    if errors:
        for error in errors:
            print(f"  - {error}")
        return 1
    states = compute_states(dag, merged)
    if as_json:
        import json

        print(json.dumps({"states": states, "next": next_per_lane(dag, merged)}, indent=2))
        return 0
    counts: dict[str, int] = {}
    for state in states.values():
        counts[state] = counts.get(state, 0) + 1
    print("states: " + ", ".join(f"{state}={count}" for state, count in sorted(counts.items())))
    for lane_id, change_id in next_per_lane(dag, merged).items():
        print(f"{lane_id:20s} {change_id or '-'}")
    return 0


def main_completion(
    change_id: str,
    dag_path: str | pathlib.Path = DAG_PATH,
    include_global: bool = True,
    dry_run: bool = False,
    merged: Sequence[str] = (),
    allow_unmerged_dependencies: bool = False,
) -> int:
    dag = load_dag(dag_path)
    try:
        plan = completion_plan(dag, change_id, include_global=include_global)
    except KeyError:
        print(f"unknown change: {change_id}")
        return 2
    commands = [plan.openspec_command, *plan.test_commands, *plan.extra_commands]
    commands.extend(plan.global_gate_commands)
    commands = [command for command in commands if command]
    pending = unmerged_dependencies(dag, change_id, merged)
    if pending and not allow_unmerged_dependencies:
        print(f"{change_id}: unmerged dependencies: {', '.join(sorted(pending))}")
        return 1
    if dry_run:
        print(f"completion plan for {change_id}:")
        for command in commands:
            print(f"  {command}")
        return 0

    failures = 0
    for command in commands:
        status, output = run_command(command)
        if (
            status != 0
            and command == plan.openspec_command
            and "not found" in output
            and changes_by_id(dag).get(change_id, {}).get("state") == "shipped"
        ):
            report(f"{command} (archived change, treated as complete)", 0, "")
            continue
        report(command, status, output, verbose=True)
        if status != 0:
            failures += 1
    if failures:
        print(f"{change_id}: {failures} completion check(s) failed")
        return 1
    print(f"{change_id}: every completion check passed")
    return 0
