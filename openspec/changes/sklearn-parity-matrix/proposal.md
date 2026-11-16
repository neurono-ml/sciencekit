# Proposal

## Why

The product goal (PRD §1.1) promises "all scikit-learn algorithms and utilities", with
every algorithm supporting datasets larger than RAM, but the implementation roadmap
(PRD §12) enumerates a small subset and the workspace currently ships **48 of the 246
estimator and transformer classes** of scikit-learn 1.9.1. There is no machine-readable
inventory: before this change the gap lived only in prose, and whole families (`feature_selection`,
`naive_bayes`, `manifold`, `mixture`, `covariance`, `cross_decomposition`,
`gaussian_process`, `neural_network`, `semi_supervised`, text vectorizers, calibration)
have no crate at all. The same audit shows that no estimator consumes
`SKMappableSource`, so the PRD §4.4 out-of-core promise is currently unfulfilled even
for the algorithms that exist.

Without a single authoritative matrix, every future change re-derives its own scope,
"complete parity" stays unfalsifiable, and structural decisions (access pattern,
sample weights, multiclass strategy) are taken by accident inside algorithm PRs.

## What Changes

- Add `parity.yaml` under this change: one row per scikit-learn 1.9.1 public estimator
  and transformer class (246), plus one row per function family (metrics, dataset
  generators, inspection utilities, preprocessing functions), with
  `status: implemented | planned | absent`, `crate`, `tier`, `change` and, for absent
  entries, a recorded `reason`.
- Add the pinned reference list `data/sklearn-1.9.1-api.txt` used to derive the matrix.
- Add the `sklearn-parity` capability spec: baseline pinning, exhaustive status
  assignment, access-pattern tier declaration per entry, cross-check against
  `dag.yaml`, and a coverage report that cannot silently shrink.
- Add a consistency checker (repo tooling, not library code) that validates the matrix:
  every entry has exactly one status, every `planned` entry names an existing change in
  `dag.yaml`, every `absent` entry carries a reason, and no `implemented` entry loses its
  coverage test.
- Extend `openspec/changes/implementation-dag/dag.yaml` with the parity program: the
  structural changes first (contracts, datasets and the sixteen shared primitives of
  `reuse-map.md`), then one change per algorithm family, in their lanes and wave order.
- Revise PRD §12 so the roadmap points at the matrix instead of enumerating a closed,
  incomplete list (the §1.1 promise and the §12 list must stop contradicting each other).
- Add a documentation page that publishes the coverage percentage and the per-family
  status, so the claim is visible and auditable by users.

No library behavior changes: this change is planning, data and tooling. Every algorithm
listed by the matrix is delivered by its own change, tracked in the DAG.

## Capabilities

### New Capabilities

- `sklearn-parity`: the normative coverage contract — pinned scikit-learn baseline,
  exhaustive one-status-per-entry matrix, access-pattern tier per entry, DAG cross-check
  and non-regressing coverage reporting.

### Modified Capabilities

None. This change does not alter estimator behavior.

## Impact

- **Planning artifacts:** new `parity.yaml`, `data/sklearn-1.9.1-api.txt`, the
  `sklearn-parity` spec, `reuse-map.md`, and the extended `implementation-dag` graph
  (~167 new changes).
- **Tooling:** one checker script and a CI invocation; no changes to `crates/`.
- **Docs:** a coverage page under `docs/src/` and the PRD §12 revision.
- **Downstream:** every algorithm change reads its matrix row to know its status, tier,
  crate and dependencies; the DAG is the scheduling source of truth for the program.
