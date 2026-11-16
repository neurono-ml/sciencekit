# Design

## Context

See `proposal.md` — Why. The source of truth is `dag.yaml` (240 nodes: 66 from the
original program, five recovered omissions, plus 168 from the scikit-learn parity
program, plus three superseded nodes kept for traceability). This design explains the layers, lanes and the automated
cycle; it must stay in sync with the YAML.

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
| L0 | representation-parity, wave-seven-deferrals, deterministic-randomness, clustering-contracts, interop-contracts, interop-foundation, gpu-compute-contracts, telemetry-export, ffi-core, sparse-pairwise-distances, kd-tree-and-ball-tree, knn-imputer, metrics-ranking, type-safe-pipeline, sparse-streaming-coverage |
| L1 | randomized-svd-kernel, r-bindings, model-selection-core, metrics-clustering, decision-tree-cart, dbscan, kmeans-lloyd-elkan, knn-classifier-regressor, svm-svc-svr, gpu-opencl-backend, gpu-cuda-backend, gpu-rocm-backend, lua-bindings, go-bindings, arrow-interop, safetensors-internal-format, json-debug-format, feature-union, pipeline-dag, streaming-pipeline |
| L2 | pca, random-forest, gradient-boosting, minibatch-kmeans-streaming, svm-linear-svc, grid-search-cv, outlier-detection, polars-interop, datafusion-interop, memory-mapped-interop, julia-dataset-interop, jvm-dataset-interop |
| L3 | truncated-svd, incremental-pca, bagging, histogram-gradient-boosting, halving-grid-search |
| L4 | model-export-retrofit (join over all model families) |
| L5 | onnx-export |
| L6 | onnx-import |
| L7 | native-model-import, python-bindings-complete |
| L8 | lora-adapters, python-dataset-interop |
| L9 | umbrella-crate |
| L10 | workflows-cli |

`polars-datafusion-sources`, `julia-bindings` and `jvm-bindings` are marked
`superseded_by` in `dag.yaml` and are skipped by the scheduler.

The full edge list lives in `dag.yaml`; the layers are a cache, not a second source of truth.

## Parallel groups

All changes in the same layer with disjoint lanes can start together. Within a lane, only
one change is `in_progress` at a time (merge-order hazard). `∥` means parallel.

| Lane | Order and parallelism |
|---|---|
| common | representation-parity → wave-seven-deferrals → deterministic-randomness → clustering-contracts → interop-contracts → interop-foundation → gpu-compute-contracts → telemetry-export |
| math | sparse-pairwise-distances ∥ randomized-svd-kernel → pairwise-metric-extension → {sparse-symmetric-eigensolver; composable-kernel-algebra} ∥ lbfgs-optimizer |
| metrics | metrics-foundation → metrics-ranking ∥ metrics-clustering → metrics-calibration-multilabel ∥ metrics-regression-completion ∥ metrics-classification-completion ∥ metrics-ranking-completion |
| neighbors | kd-tree-and-ball-tree ∥ knn-imputer ∥ svm-svc-svr → knn-classifier-regressor → svm-linear-svc → {radius-neighbors; neighbor-transformers; kernel-density; nearest-centroid; neighborhood-components-analysis} |
| tree | decision-tree-cart → {random-forest → bagging; gradient-boosting → histogram-gradient-boosting; outlier-detection} |
| cluster | kmeans-lloyd-elkan → minibatch-kmeans-streaming; dbscan → agglomerative-clustering → {feature-agglomeration; spectral-clustering → {spectral-biclustering; spectral-coclustering}}; affinity-propagation ∥ mean-shift ∥ birch ∥ bisecting-kmeans ∥ optics → hdbscan |
| decomposition | randomized-svd-kernel → pca → {truncated-svd; incremental-pca}; nmf → minibatch-nmf ∥ sparse-pca → minibatch-sparse-pca ∥ dictionary-learning → {minibatch-dictionary-learning; sparse-coder} ∥ fast-ica ∥ factor-analysis ∥ latent-dirichlet-allocation ∥ random-projection |
| model-selection | model-selection-core → grid-search-cv → halving-grid-search; parameter-grid → randomized-search → halving-random-search; {leave-one-out-splitters; shuffle-splitters; repeated-kfold; stratified-group-kfold; time-series-split; predefined-split} ∥ cross-validate-functions → {learning-curve; validation-curve; permutation-test-score}; threshold-tuning |
| pipeline | type-safe-pipeline → {feature-union; pipeline-dag; streaming-pipeline; column-transformer; transformed-target-regressor} |
| interop | interop-foundation → arrow-interop → {polars-interop; datafusion-interop; memory-mapped-interop} → {safetensors-internal-format; json-debug-format} → model-export-retrofit → onnx-export → onnx-import → native-model-import → lora-adapters |
| gpu | gpu-compute-contracts → {gpu-opencl-backend; gpu-cuda-backend → deferred; gpu-rocm-backend → deferred} |
| bindings | ffi-core → extract-language-bindings → {lua-bindings; go-bindings} |
| python | python-bindings-complete → python-dataset-interop |
| julia | julia-dataset-interop |
| jvm | jvm-dataset-interop |
| r | r-bindings |
| streaming | streaming-source-provider → sparse-streaming-batches → sparse-streaming-coverage |
| product | umbrella-crate → workflows-cli |
| planning | sklearn-parity-matrix |
| contracts | access-pattern-contract → structured-tracing → {mappable-fit-path ∥ reopenable-fit-path} → sample-weight-support → estimator-tags-and-callbacks → {label-and-indicator-primitives → boxed-predictor-contract → multiclass-strategy → multioutput-strategy}; resampling-primitives ∥ frozen-estimator |
| datasets | dataset-generators → dataset-loaders |
| preprocessing | max-abs-scaler → normalizer → binarizer → function-transformer → kernel-centerer → label-encoder → label-binarizer → multi-label-binarizer → ordinal-encoder → k-bins-discretizer → power-transformer → quantile-transformer → spline-transformer → target-encoder → preprocessing-functions |
| feature-selection | variance-threshold ∥ univariate-score-functions → univariate-selectors; select-from-model; rfe → sequential-feature-selector |
| naive-bayes | gaussian ∥ multinomial ∥ complement ∥ bernoulli ∥ categorical |
| linear-model | sgd-classifier-regressor → penalized-glm-solver → regularization-path-solver → ridge-classifier → ridge-cv → lasso-cv → elastic-net-cv → logistic-regression-cv → multitask-linear → lars-family → orthogonal-matching-pursuit → bayesian-ridge → ard-regression → generalized-linear-regressors → quantile-regressor → huber-regressor → ransac-regressor → theil-sen-regressor → passive-aggressive → perceptron → sgd-one-class-svm |
| kernel | pairwise-kernel-functions → nusvm ∥ linear-svr ∥ kernel-ridge ∥ kernel-approximation ∥ gaussian-process ∥ kernel-pca |
| neural-network | multilayer-perceptron ∥ bernoulli-rbm |
| semi-supervised | label-propagation ∥ label-spreading ∥ self-training |
| text | text-tokenization-and-vocabulary → {text-vectorizers; feature-hasher}; dict-vectorizer |
| ensemble | extra-trees ∥ adaboost ∥ voting ∥ stacking ∥ random-trees-embedding |
| meta | dummy-estimators ∥ isotonic-regression |
| calibration | calibrated-classifier |
| inspection | permutation-importance ∥ partial-dependence |
| mixture | gaussian-mixture → bayesian-gaussian-mixture |
| discriminant | linear-discriminant-analysis ∥ quadratic-discriminant-analysis |
| covariance | empirical-covariance → {shrunk-covariance → {ledoit-wolf; oas}; min-cov-det → elliptic-envelope; graphical-lasso} |
| cross-decomposition | pls-svd → {pls-regression; pls-canonical; canonical-correlation-analysis} |
| manifold | multidimensional-scaling → classical-mds; isomap ∥ locally-linear-embedding ∥ spectral-embedding ∥ tsne |
| stats | streaming-moments → {distribution-and-em-kernels; robust-statistics-kernels; binning-primitives} |
| graph | graph-construction-and-algorithms |
| impute | iterative-imputer |
| image | image-patch-extraction |

## Parity program (scikit-learn 1.9.1)

The parity program is a separate wave sequence appended to the same graph; the matrix in
`openspec/changes/sklearn-parity-matrix/parity.yaml` is its inventory and the DAG is its
schedule. The granularity rule is one change per algorithm, with variants that share a
solver grouped in that algorithm's change (see the parity design).

| Wave | Content | Hard structural dependencies |
|---|---|---|
| 8 | `sklearn-parity-matrix`, the eight contract changes, `dataset-generators`/`dataset-loaders`, and the sixteen shared primitives from `sklearn-parity-matrix/reuse-map.md` (`streaming-moments`, `distribution-and-em-kernels`, `robust-statistics-kernels`, `binning-primitives`, `pairwise-metric-extension`, `sparse-symmetric-eigensolver`, `composable-kernel-algebra`, `lbfgs-optimizer`, `graph-construction-and-algorithms`, `penalized-glm-solver`, `regularization-path-solver`, `matrix-factorization-kernels`, `resampling-primitives`, `boxed-predictor-contract`, `label-and-indicator-primitives`, `text-tokenization-and-vocabulary`) | none |
| 9 | preprocessing completion (15), feature selection (6), naive Bayes (5), dummy and isotonic (2), metrics completion (3) | access pattern, sample weights |
| 10 | linear-model completion (18), model-selection completion (14) | wave 9 encoders, model-selection-core |
| 11 | kernel and GP (7), neural networks (2), semi-supervised (3), text vectorizers (3), compose (2), ensemble extras (5), calibration (1), inspection (2) | wave 10 selection |
| 12 | clustering (11), decomposition (11), mixtures (2), discriminant (2), covariance (7), cross-decomposition (4) | distances, SVD, graph nodes |
| 13 | manifold (6), remaining neighbors (5) | wave 12 |

Every algorithm change depends on `access-pattern-contract`; weighted estimators also
depend on `sample-weight-support`, binary-only classifiers on `multiclass-strategy`, and
each family depends on the shared primitives listed in `reuse-map.md` (the DAG edge is the
enforcement; the map is the explanation). Primitives are foundations: they may only change
through their own change, and algorithm changes consume them read-only.

## Node states

Every node carries `state`:

| State | Meaning |
|---|---|
| `shipped` | code and openspec PRs merged into `main`; not schedulable |
| `closing` | implementation merged; the remaining task is the ADR, review or archive |
| `in-progress` | tasks open, not blocked |
| `planned` | not started |
| `deferred` | intentionally postponed with a recorded reason |
| `superseded` | replaced by another node, kept for traceability |

The readiness script filters `shipped` and `superseded` out of the ready set.

## Execution waves

The waves are the scheduling sequence toward the target; `parallel: false` waves run one
change at a time. The wave list is a cache computed from `dag.yaml` dependencies and
lane capacities; `dag.yaml` remains the source of truth.

| Wave | Goal | Parallel |
|---|---|---|
| W0 close-in-flight | close the nineteen `closing` changes (ADR + PR; book chapters deferred) | per lane |
| W1 dataset-interop | Polars, DataFusion, Julia dataset paths (JVM stays deferred) | yes |
| W2 foundations | parity matrix, contracts, structured tracing, the shared primitives | yes, per lane |
| W3 model-selection-and-metrics | selection utilities, curves, calibration, outlier detectors | yes |
| W4 preprocessing-selection-bayes | preprocessing completion, feature selection, naive Bayes, meta baselines | yes |
| W5 linear-models | the remaining linear family | yes |
| W6 kernel-neural-text-ensemble | kernel methods, GP, neural, semi-supervised, text, compose, ensembles | yes |
| W7 unsupervised | clustering, decomposition, mixtures, discriminant, covariance, PLS/CCA | yes |
| W8 manifold-and-neighbors | manifold learning, remaining neighbors | yes |
| W8.5 gpu-accelerators | CUDA and ROCm backends | yes |
| W9 joins | umbrella crate, Python bindings, CLI | no |

## Constitution

The immutable clauses (cláusulas pétreas) every change must keep, recorded in
`dag.yaml` under `policy.immutable_clauses`: full 1.9.1 parity with the access-pattern
doctrine; builder pattern with `backend` and `execution_mode`; naming rules; zero-copy
and representation parity without densification; TDD with one commit per task; the
worktree and dual-PR workflow; an ADR per change; the changelog; and the tracing policy.

## Downstream mutability

Later waves are **not** immutable: implementing an earlier wave may amend a later change
(proposal, design, tasks, dependencies) when the implementation evidence demands it, as
long as every immutable clause still holds. The amendment is recorded in the later change
and in its ADR. This is the rule that lets Wave 2 contracts shape Waves 3–8 without a
big-bang replan.

## Documentation policy

Book chapters (`docs/src/development-explained/`) are deferred for the whole program:
the book's format is being redesigned and will be rebuilt from the ADRs. What stays
mandatory per shipped change: the ADR issue (new or amended), the changelog entry, the
API docs and the coverage page. The retained set is listed in
`policy.documentation_policy.retained`.

## Acceptance and tracing policies

Every algorithm change's completion includes, per `policy.acceptance_policy`:
end-to-end tests with and without lazy/out-of-core inputs, memory and resource guard
tests, export/reload round-trip where the family is exportable, and large/small data
fixtures. Per `policy.tracing_policy`: nested spans (pipeline > stage > operation > step
> epoch/batch), a trace per inference, and one span vocabulary, enforced by the
`structured-tracing` conformance test.

## Joins and splits

- `model-export-retrofit` is a **join**: it needs every fitted model family. At apply time
  its task groups can be split into per-family sub-branches (preprocessing, linear,
  neighbors/SVM, tree/ensemble, unsupervised, pipeline) that start as each family lands;
  the change completes only when all sub-branches merge.
- `interop-foundation` creates the `sciencekit_interop` crate; `arrow-interop` is the
  first concrete adapter and unblocks `polars-interop`, `datafusion-interop`,
  `memory-mapped-interop` and the Python/Julia/JVM dataset changes.
- `python-dataset-interop` extends `python-bindings-complete`; it is a join over the
  Arrow/host-language dataset paths but can start once its hard dependencies merge.
- `julia-dataset-interop` and `jvm-dataset-interop` supersede the earlier
  `julia-bindings` and `jvm-bindings`; the older nodes stay in the graph as
  `status: superseded` and are never scheduled.
- `polars-datafusion-sources` is superseded by `arrow-interop`, `polars-interop`,
  `datafusion-interop` and `memory-mapped-interop`; it is not scheduled.
- `r-bindings` is the R lane over the stable C ABI: it needs `ffi-core` for the ABI
  itself and `interop-foundation` for the model-state contract behind save/load.
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
