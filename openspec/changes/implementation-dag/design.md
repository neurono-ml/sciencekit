# Design

## Context

See `proposal.md` — Why. The source of truth is `dag.yaml` (55 active changes). This design
explains the layers, lanes and the automated cycle; it must stay in sync with the YAML.

## Goals / Non-Goals

**Goals:**
- A ready/blocked computation agents can follow mechanically.
- Maximum safe parallelism without crate-level edit conflicts.
- A completion definition that includes tests, docs, review and PR merge.

**Non-Goals:**
- Replacing OpenSpec status/tasks.
- Scheduling CI infrastructure.
- Application code.

## Layers

Computed from `depends_on` only (soft dependencies do not block start):

| Layer | Changes ready at this layer |
|---|---|
| L0 | representation-parity, wave-seven-deferrals, deterministic-randomness, clustering-contracts, interop-contracts, gpu-compute-contracts, telemetry-export, ffi-core, sparse-pairwise-distances, kd-tree-and-ball-tree, knn-imputer, metrics-ranking, type-safe-pipeline, sparse-streaming-coverage |
| L1 | randomized-svd-kernel, model-selection-core, metrics-clustering, decision-tree-cart, dbscan, kmeans-lloyd-elkan, knn-classifier-regressor, svm-svc-svr, gpu-opencl-backend, gpu-cuda-backend, gpu-rocm-backend, lua-bindings, julia-bindings, go-bindings, jvm-bindings, safetensors-internal-format, json-debug-format, polars-datafusion-sources, feature-union, pipeline-dag, streaming-pipeline |
| L2 | pca, random-forest, gradient-boosting, minibatch-kmeans-streaming, svm-linear-svc, grid-search-cv, outlier-detection |
| L3 | truncated-svd, incremental-pca, bagging, histogram-gradient-boosting, halving-grid-search |
| L4 | model-export-retrofit (join over all model families) |
| L5 | onnx-export |
| L6 | onnx-import |
| L7 | native-model-import, python-bindings-complete |
| L8 | lora-adapters |
| L9 | umbrella-crate |
| L10 | workflows-cli |

The full edge list lives in `dag.yaml`; the layers are a cache, not a second source of truth.

## Parallel groups

All changes in the same layer with disjoint lanes can start together. Within a lane, only
one change is `in_progress` at a time (merge-order hazard). `∥` means parallel.

| Lane | Order and parallelism |
|---|---|
| common | representation-parity → wave-seven-deferrals → deterministic-randomness → clustering-contracts → interop-contracts → gpu-compute-contracts → telemetry-export |
| math | sparse-pairwise-distances ∥ randomized-svd-kernel |
| metrics | metrics-ranking ∥ metrics-clustering |
| neighbors | kd-tree-and-ball-tree ∥ knn-imputer ∥ svm-svc-svr → knn-classifier-regressor → svm-linear-svc |
| tree | decision-tree-cart → {random-forest → bagging; gradient-boosting → histogram-gradient-boosting; outlier-detection} |
| cluster | kmeans-lloyd-elkan → minibatch-kmeans-streaming; dbscan |
| decomposition | randomized-svd-kernel → pca → {truncated-svd; incremental-pca} |
| model-selection | model-selection-core → grid-search-cv → halving-grid-search |
| pipeline | type-safe-pipeline → {feature-union; pipeline-dag; streaming-pipeline} |
| interop | interop-contracts → {safetensors-internal-format; json-debug-format; polars-datafusion-sources} → model-export-retrofit → onnx-export → onnx-import → native-model-import → lora-adapters |
| gpu | gpu-compute-contracts → {gpu-opencl-backend; gpu-cuda-backend; gpu-rocm-backend} |
| bindings | ffi-core → {lua-bindings; julia-bindings; go-bindings; jvm-bindings} |
| streaming | sparse-streaming-coverage |
| product | umbrella-crate → {python-bindings-complete; workflows-cli} |

## Joins and splits

- `model-export-retrofit` is a **join**: it needs every fitted model family. At apply time
  its task groups can be split into per-family sub-branches (preprocessing, linear,
  neighbors/SVM, tree/ensemble, unsupervised, pipeline) that start as each family lands;
  the change completes only when all sub-branches merge.
- `umbrella-crate`, `python-bindings-complete` and `workflows-cli` are joins and should
  start only after their dependency sets are merged.
- `safetensors-internal-format` and `json-debug-format` have a **soft** dependency on
  `model-export-retrofit`: they can start with mock states after `interop-contracts`, but
  their integration tests require the retrofit.

## Completion checks

A change is `done` only when all of the following hold:

1. `openspec status --change <id> --json` reports every artifact `done` (or deliberately
   skipped) and all tasks checked.
2. `openspec validate <id>` is green.
3. The change-specific tests in `dag.yaml` are green.
4. The global gates are green:
   ```sh
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
5. Docs build when the change ships docs: `mdbook build docs`.
6. Independent review passed and findings addressed.
7. Code PR merged into `main`; openspec PR merged into `main`; docs PR merged into
   `docs/documentations`.
8. No unmerged dependency remains in `depends_on`.

`soft_depends_on` entries do not block start, but they must be merged before the change is
marked `done`.

## Automated cycle

```
loop:
  graph = load dag.yaml
  for change in graph.changes:
    if change.pr_merged: state = done
    elif all(dep in done for dep in change.depends_on):
      state = ready
    else: state = blocked
  for lane in graph.lanes:
    ready_lane = [c for c in ready if c.lane == lane.id]
    if ready_lane and lane.in_progress is empty:
      launch one agent with c (prefer the lane merge_order)
  for finished change:
    verify completion checks
    merge code PR, openspec PR and docs PR
    mark done; recompute ready
```

Guardrails:

- One in-progress change per lane; different lanes run in parallel.
- Rebase before merge; never commit directly to `main`.
- A change whose tests fail stays `in_progress`; downstream changes stay `blocked`.
- Docs/spec-only changes (`representation-parity`, `wave-seven-deferrals`,
  `implementation-dag`) are `done` when their PRs merge and `mdbook build docs` is green.
- Hardware-gated GPU tests may be skipped when no device is present; fallback tests must
  pass.

## Risks / Trade-offs

- **Lane serialization reduces parallelism** → it prevents merge conflicts in shared
  crates; the parallel groups are still broad.
- **Join tail** → `model-export-retrofit`/`umbrella`/`python`/`cli` are late; split the
  retrofit by family and start the umbrella with core crates if needed.
- **Stale layers** → `dag.yaml` is the source of truth; layers are documentation.
- **Completion drift** → checks are explicit; a change is not done until its PRs merge.

## Open Questions

None blocking. The readiness script is planned in `tasks.md`.
