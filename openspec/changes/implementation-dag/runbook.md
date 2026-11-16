# Implementation DAG runbook

The exact loop an agent follows to burn the graph down. `dag.yaml` is the source of
truth; this runbook only spells out the commands.

## 1. Sync and recompute readiness

```sh
git fetch --prune origin
git checkout main && git pull --ff-only
git checkout docs/documentations && git pull --ff-only
git checkout main

python3 scripts/check-dag.py                 # graph consistency, fast
python3 scripts/dag-readiness.py             # done/ready/blocked + next per lane
```

`dag-readiness.py` treats a change as `done` when its state is `shipped`, when it is
`superseded`, or when its id is passed with `--merged <id>` / `--merged-file <path>`.
`deferred` changes are never scheduled. A lane whose merge order still holds an
`in-progress` or `closing` change reports no next change: the lane is occupied.

**The `next` column is the only concurrency authority.** The `states` map is a raw
dependency computation (several changes can be `ready` at once); scheduling from it can
double-book a lane. Always start the change named in `next`.

## 2. Pick the next change

The graph is regenerated documentation as well: `python3 scripts/check-dag.py` also fails
when the generated layer and lane blocks in `design.md` drift from `dag.yaml`; refresh them
with `python3 scripts/render-dag-design.py`.

Walk the `next` column of `dag-readiness.py` in wave order (`execution_waves` in
`dag.yaml`). `next` already enforces the lane prefix: if an earlier `merge_order` member
is still blocked, the lane reports `-` and must wait. Start at most one change per lane.

Wave 0 (`closing` changes) is the exception: those lanes are occupied by definition, so
`next` never names them. Pick a W0 change from the `closing` set in `dag.yaml` and finish
its remaining review/archive task; its lane is already held by that same change.

## 3. Announce the change and create the worktrees

```sh
python3 scripts/dag-readiness.py --json | jq '.next'   # is <change> the lane's next?
git worktree add temporary/worktrees/feat/<change> -b feat/<change> main
git worktree add temporary/worktrees/feat/<change>-openspec \
  -b feat/<change>-openspec main
```

Open the ADR issue in English (`architecture-decision-records` skill: Y-statement for a
simple decision, Nygard for a complete one) and keep its number: both PRs close it. The
change enters `in-progress` in `dag.yaml` on the openspec branch.

## 4. Apply with TDD

```sh
openspec status --change <change> --json
openspec instructions apply --change <change> --json
```

Read every `contextFiles` entry, then implement task by task: write the failing test
first, confirm the failure, implement the minimum that passes, refactor. One commit per
task on the code branch; mark the same task `- [x]` on the openspec branch in the same
step.

## 5. Gates and completion

```sh
python3 scripts/check-completion.py --change <change>
```

That runs, in order: the OpenSpec status command, every `completion.tests` entry, every
`completion.extra` entry (for example `openspec validate <change>` and
`python3 scripts/check-parity-matrix.py`), the global gates (`cargo fmt --all --check`, `cargo clippy --workspace
--all-targets -- -D warnings`, `cargo test --workspace`) and `mdbook build docs` when the
change lists it. `--dry-run` prints the plan; `--skip-global` runs only the change's own
checks while iterating. A failing check keeps the change `in-progress`.

Also run `openspec validate <change>` and the change's acceptance checks from
`policy.acceptance_policy` (lazy and non-lazy end-to-end, memory guards, export/reload).

## 6. Independent review

Launch a separate agent with the change's `proposal.md`, `tasks.md`, the diff and the gate
output. It must confirm the spec is met, the immutable clauses hold and the tests prove
it. Record the verdict on the openspec branch.

## 7. Push, PRs and merge

```sh
git push -u origin feat/<change>
git push -u origin feat/<change>-openspec
gh pr create --base main --head feat/<change> ... --body "Closes #<adr>"
gh pr create --base main --head feat/<change>-openspec ... --body "Closes #<adr>"
```

A shipped docs change also opens a docs PR against `docs/documentations`. Rebase before
merge. Merge the code PR and the openspec PR in any order, then update `CHANGELOG.md`
(Keep a Changelog 1.1.0), mark the node `shipped` in `dag.yaml`, archive the OpenSpec
change, delete the merged branches and remove the worktrees.

## 8. Recompute

Back on `main` after the merge:

```sh
python3 scripts/check-dag.py
python3 scripts/dag-readiness.py
```

The finished change disappears from the ready set and unblocks its downstream nodes.

## Guardrails

- One in-progress change per lane; different lanes run in parallel.
- Never commit on `main`; every change lands through its code and openspec PRs.
- A change whose checks fail stays `in-progress`; downstream stays `blocked`.
- Docs/spec-only changes are done when their PRs merge and `mdbook build docs` is green.
- `soft_depends_on` does not block start but must be merged before the change is `done`.
- Hardware-gated GPU tests may skip without a device; fallback tests must pass.
- Downstream changes stay mutable: implementation evidence may amend a later change's
  proposal, design, tasks or dependencies while every immutable clause still holds; the
  amendment is recorded in that change and its ADR.
