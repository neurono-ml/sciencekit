# Proposal

## Why

There are 240 graph nodes, built in layers: the original program, then the scikit-learn
parity program (Waves 8–13, including sixteen shared primitives and the
`reopenable-fit-path` retrofit), plus three superseded nodes kept for traceability. The
exact set and the authoritative breakdown by state are in `dag.yaml` and restated in
`design.md`; the counts in the historical task steps are point-in-time and do not sum to
the current total. Agents need one authoritative dependency graph, with
completion checks, so they can start independent changes in parallel, block on real
dependencies, and avoid merge conflicts in shared crates. Today the ordering is spread
across wave-plan prose and per-change designs.

## What Changes

- Add `openspec/changes/implementation-dag/dag.yaml`: a machine-readable graph of every
  active change with `depends_on`, optional `soft_depends_on`, lane, join semantics and
  completion checks.
- Add a human-readable design with the layered view, parallel groups, merge lanes and the
  automated execution loop.
- Define completion checks: OpenSpec status, package/language tests, workspace gates,
  docs build, independent review and PRs merged.
- Define lane capacities so two agents never edit the same crate concurrently.
- Document join nodes (`model-export-retrofit`, `umbrella-crate`,
  `python-bindings-complete`, `workflows-cli`) and how they can be split into family-level
  work.
- Dataset interop nodes included: `interop-foundation`, `arrow-interop`,
  `polars-interop`, `datafusion-interop`, `memory-mapped-interop`,
  `python-dataset-interop`, `julia-dataset-interop`, `jvm-dataset-interop`.
- Language-binding nodes included: `lua-bindings`, `go-bindings`, `r-bindings` (own
  `r` lane, alongside the `python`, `julia` and `jvm` lanes).
- Superseded nodes retained for traceability: `polars-datafusion-sources`,
  `julia-bindings`, `jvm-bindings`.
- Extend the graph with the scikit-learn 1.9.1 parity program: structural contracts and
  shared primitives first, then one change per algorithm family in Waves 8–13, with the
  inventory owned by `sklearn-parity-matrix/parity.yaml` and the reuse map by
  `sklearn-parity-matrix/reuse-map.md`.
- Mark every node with `state` (shipped, closing, in-progress, planned, deferred,
  superseded) and add the execution waves (W0–W9) that order the program toward the
  target: every parity change shipped.
- Record the immutable clauses, the downstream-mutability rule, the documentation policy
  (book chapters deferred; one ADR per shipped change), the acceptance policy (end-to-end
  tests with and without lazy inputs, memory guards, export/reload) and the tracing policy
  (nested spans, a trace per inference) in the DAG policy block.
- Add the `structured-tracing` change (span hierarchy: pipeline > stage > operation >
  step > epoch/batch) to the contracts lane.
- No application code and no runtime behavior change; `.openspec.yaml` sets
  `skip_specs: true`.

## Capabilities

### New Capabilities

None. This is a planning/coordination artifact; no spec-level behavior changes.

## Impact

- **Docs/specs:** new `dag.yaml`, `design.md` and tasks under
  `openspec/changes/implementation-dag/`.
- **Downstream:** every future agent reads `dag.yaml` to compute ready/blocked changes.
- **Acceptance criteria:** the YAML parses, every dependency resolves to a known change,
  every change has a lane and completion checks, and the design matches the YAML.
