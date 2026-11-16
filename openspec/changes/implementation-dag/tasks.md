# Tasks

## 1. Graph

- [x] 1.1 Add `dag.yaml` with every active change, lane, `depends_on`,
  `soft_depends_on`, join flags and completion checks; verify it parses as YAML.
- [x] 1.2 Add a consistency check that every dependency resolves to a known change, there
  are no duplicates and every change has a lane/completion block; verify with the check.
- [x] 1.3 Add the lane `merge_order` lists and confirm they match the design table.
- [x] 1.5 Add the scikit-learn 1.9.1 parity program to `dag.yaml`: 146 changes in Waves
  8–13, twenty-three new lanes, the `planning`/`contracts` structural changes first, and the
  extended lane merge orders; verify the graph parses and every dependency resolves.
- [x] 1.6 Add the sixteen shared primitives and `iterative-imputer` (163 parity nodes
  total), rewire 104 algorithm changes onto them, and add the `stats`, `graph` and
  `impute` lanes; verify no dependency is unresolved and same-lane ordering holds.
- [x] 1.7 Close the matrix audit gaps: `extremely-randomized-tree`,
  `group-shuffle-split`, `image-patch-extraction`, `tree-text-export` (167 parity nodes
  total, 233 graph nodes); verify every `planned` matrix row resolves to a graph node.
- [x] 1.8 Close the conformance-audit gaps on the implemented estimators: add
  `reopenable-fit-path` (iterative `fit_streaming`) to the contracts lane, and rewire
  `outlier-lof-one-class` and `metrics-calibration-multilabel` onto the contracts they
  need (234 graph nodes); verify no dependency is unresolved.
- [x] 1.4 Add the `r` language lane with the `r-bindings` change (own repository
  `neurono-ml/sciencekit-r`, depending on `ffi-core` and `interop-foundation`); verify
  the graph still parses and every dependency resolves.

- [x] 1.9 Mark every node with `state` (shipped, closing, in-progress, planned, deferred,
  superseded) from the merge state and the openspec status; verify the counts
  (40 shipped, 19 closing, 3 superseded, 3 deferred, 1 in-progress, 174 planned) and that
  the readiness set excludes shipped, superseded and deferred nodes.
- [x] 1.10 Add the execution waves W0–W9 (close-in-flight, dataset interop, foundations,
  model selection, preprocessing/selection/Bayes, linear models, kernel/neural/text,
  unsupervised, manifold/neighbors, GPU accelerators, joins) with per-wave goals and
  parallelism; verify every active node belongs to exactly one wave.
- [x] 1.11 Record the immutable clauses, the downstream-mutability rule, the
  documentation policy (book deferred, ADR per change), the acceptance policy (end-to-end
  lazy and non-lazy tests, memory guards, export/reload) and the tracing policy in
  `policy`; verify the policy block parses.
- [x] 1.12 Add the `structured-tracing` node to the contracts lane and to Wave 2, with its
  completion tests; verify the graph parses and every dependency resolves.

## 2. Readiness and completion tooling

- [x] 2.1 Add a readiness script that reads `dag.yaml`, marks each change
  `done`/`ready`/`blocked` from a merged set, and prints the next runnable change per lane;
  verify it with a fixture and the real graph.
- [x] 2.2 Add a completion checker that runs the change's `completion.tests` plus the
  global gates and reports pass/fail; verify on a small change.
- [x] 2.3 Document the exact agent loop (pick ready lane → apply → gates → review → PRs →
  merge → recompute) and the guardrails.

## 3. Documentation and review

- [x] 3.1 Keep `design.md` layers/parallel tables in sync with `dag.yaml`; verify with the
  consistency check.
- [x] 3.2 Run `openspec validate implementation-dag` and `mdbook build docs`.
- [ ] 3.3 Independent review by a separate agent validating that every active change is in
  the graph, every dependency is real and the parallel groups do not share a lane.

## Workflow follow-up

- Create the matching GitHub issue as an ADR in English and reference it from both PRs
  with a closing keyword.
- Push the docs branch and the `-openspec` branch and open both PRs.
- Archive the change after both PRs merge, then delete the merged branches and worktrees.
