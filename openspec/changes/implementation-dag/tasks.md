# Tasks

## 1. Graph

- [ ] 1.1 Add `dag.yaml` with every active change, lane, `depends_on`,
  `soft_depends_on`, join flags and completion checks; verify it parses as YAML.
- [ ] 1.2 Add a consistency check that every dependency resolves to a known change, there
  are no duplicates and every change has a lane/completion block; verify with the check.
- [ ] 1.3 Add the lane `merge_order` lists and confirm they match the design table.

## 2. Readiness and completion tooling

- [ ] 2.1 Add a readiness script that reads `dag.yaml`, marks each change
  `done`/`ready`/`blocked` from a merged set, and prints the next runnable change per lane;
  verify it with a fixture and the real graph.
- [ ] 2.2 Add a completion checker that runs the change's `completion.tests` plus the
  global gates and reports pass/fail; verify on a small change.
- [ ] 2.3 Document the exact agent loop (pick ready lane → apply → gates → review → PRs →
  merge → recompute) and the guardrails.

## 3. Documentation and review

- [ ] 3.1 Keep `design.md` layers/parallel tables in sync with `dag.yaml`; verify with the
  consistency check.
- [ ] 3.2 Run `openspec validate implementation-dag` and `mdbook build docs`.
- [ ] 3.3 Independent review by a separate agent validating that every active change is in
  the graph, every dependency is real and the parallel groups do not share a lane.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
