# Design

## Context

See `proposal.md` — Why. The source of truth is `dag.yaml`: 240 nodes = 40 shipped, 19
closing, 3 superseded, 3 deferred, 1 in-progress, 174 planned. This design explains the
layers, lanes and the automated cycle; the layer and lane blocks are generated from the
YAML and `scripts/check-dag.py` fails when they drift.

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

Computed from `depends_on` only (soft dependencies do not block start; superseded nodes are
excluded). The block below is generated from `dag.yaml`; a hand edit fails
`scripts/check-dag.py`.

<!-- dag-layers:start -->
| Layer | Changes |
|---|---|
| L0 | clustering-contracts, deterministic-randomness, ffi-core, gpu-compute-contracts, interop-contracts, interop-foundation, kd-tree-and-ball-tree, knn-imputer, label-and-indicator-primitives, lbfgs-optimizer, metrics-foundation, metrics-ranking, representation-parity, sklearn-parity-matrix, sparse-pairwise-distances, sparse-streaming-coverage, streaming-source-provider, telemetry-export, type-safe-pipeline, wave-seven-deferrals |
| L1 | access-pattern-contract, arrow-interop, dbscan, decision-tree-cart, estimator-tags-and-callbacks, extract-language-bindings, feature-union, go-bindings, gpu-cuda-backend, gpu-opencl-backend, gpu-rocm-backend, json-debug-format, kmeans-lloyd-elkan, knn-classifier-regressor, lua-bindings, metrics-clustering, model-selection-core, pairwise-metric-extension, pipeline-dag, r-bindings, randomized-svd-kernel, resampling-primitives, safetensors-internal-format, sample-weight-support, sgd-classifier-regressor, sparse-streaming-batches, streaming-pipeline, svm-svc-svr, text-tokenization-and-vocabulary |
| L2 | adaboost, affinity-propagation, agglomerative-clustering, ard-regression, bayesian-ridge, bernoulli-rbm, binarizer, bisecting-kmeans, boxed-predictor-contract, composable-kernel-algebra, cross-validate-functions, datafusion-interop, dataset-generators, dict-vectorizer, extremely-randomized-tree, feature-hasher, frozen-estimator, function-transformer, gradient-boosting, graph-construction-and-algorithms, grid-search-cv, group-shuffle-split, image-patch-extraction, isotonic-regression, julia-dataset-interop, jvm-dataset-interop, kernel-centerer, kernel-density, label-binarizer, label-encoder, leave-one-out-splitters, matrix-factorization-kernels, max-abs-scaler, mean-shift, memory-mapped-interop, metrics-classification-completion, metrics-regression-completion, minibatch-kmeans-streaming, multi-label-binarizer, multiclass-strategy, multilayer-perceptron, neighbor-transformers, neighborhood-components-analysis, normalizer, nusvm, ordinal-encoder, outlier-detection, pairwise-kernel-functions, parameter-grid, pca, polars-interop, predefined-split, preprocessing-functions, radius-neighbors, random-forest, random-projection, reopenable-fit-path, repeated-kfold, ridge-cv, sgd-one-class-svm, shuffle-splitters, sparse-symmetric-eigensolver, spline-transformer, stratified-group-kfold, streaming-moments, structured-tracing, svm-linear-svc, text-vectorizers, time-series-split, tree-text-export |
| L3 | bagging, binning-primitives, birch, column-transformer, dataset-loaders, dictionary-learning, distribution-and-em-kernels, dummy-estimators, empirical-covariance, extra-trees, fast-ica, feature-agglomeration, gaussian-process, halving-grid-search, hdbscan, histogram-gradient-boosting, incremental-pca, isomap, iterative-imputer, kernel-approximation, kernel-pca, kernel-ridge, label-propagation, label-spreading, learning-curve, linear-discriminant-analysis, linear-svr, locally-linear-embedding, mappable-fit-path, multidimensional-scaling, multioutput-strategy, nearest-centroid, nmf, optics, passive-aggressive, penalized-glm-solver, perceptron, permutation-importance, permutation-test-score, pls-svd, power-transformer, quadratic-discriminant-analysis, random-trees-embedding, randomized-search, regularization-path-solver, rfe, ridge-classifier, robust-statistics-kernels, select-from-model, self-training, sequential-feature-selector, sparse-pca, spectral-clustering, spectral-embedding, stacking, target-encoder, transformed-target-regressor, truncated-svd, tsne, univariate-score-functions, validation-curve, variance-threshold, voting |
| L4 | bernoulli-naive-bayes, calibrated-classifier, canonical-correlation-analysis, categorical-naive-bayes, classical-mds, complement-naive-bayes, elastic-net-cv, factor-analysis, gaussian-mixture, gaussian-naive-bayes, generalized-linear-regressors, graphical-lasso, halving-random-search, huber-regressor, k-bins-discretizer, lars-family, lasso-cv, latent-dirichlet-allocation, logistic-regression-cv, metrics-calibration-multilabel, min-cov-det, minibatch-dictionary-learning, minibatch-nmf, minibatch-sparse-pca, model-export-retrofit, multinomial-naive-bayes, multitask-linear, orthogonal-matching-pursuit, outlier-lof-one-class, partial-dependence, pls-canonical, pls-regression, quantile-regressor, quantile-transformer, ransac-regressor, shrunk-covariance, sparse-coder, spectral-biclustering, spectral-coclustering, theil-sen-regressor, threshold-tuning, univariate-selectors |
| L5 | bayesian-gaussian-mixture, elliptic-envelope, ledoit-wolf, metrics-ranking-completion, oas, onnx-export |
| L6 | onnx-import |
| L7 | native-model-import, python-bindings-complete |
| L8 | lora-adapters, python-dataset-interop |
| L9 | umbrella-crate |
| L10 | workflows-cli |
<!-- dag-layers:end -->

## Parallel groups

All changes in the same layer with disjoint lanes can start together. Within a lane only
one change is `in-progress` at a time and the lane order is the `merge_order` below; the
block is generated from `dag.yaml` and a hand edit fails `scripts/check-dag.py`.

<!-- dag-lanes:start -->
| Lane | Merge order |
|---|---|
| common | representation-parity → wave-seven-deferrals → deterministic-randomness → clustering-contracts → interop-contracts → interop-foundation → gpu-compute-contracts → telemetry-export |
| math | sparse-pairwise-distances → randomized-svd-kernel → pairwise-metric-extension → sparse-symmetric-eigensolver → composable-kernel-algebra → lbfgs-optimizer |
| metrics | metrics-foundation → metrics-ranking → metrics-clustering → metrics-calibration-multilabel → metrics-regression-completion → metrics-classification-completion → metrics-ranking-completion |
| tree | decision-tree-cart → extremely-randomized-tree → tree-text-export → random-forest → bagging → gradient-boosting → histogram-gradient-boosting → outlier-detection → outlier-lof-one-class |
| neighbors | kd-tree-and-ball-tree → knn-classifier-regressor → knn-imputer → svm-svc-svr → svm-linear-svc → radius-neighbors → neighbor-transformers → kernel-density → nearest-centroid → neighborhood-components-analysis |
| cluster | kmeans-lloyd-elkan → minibatch-kmeans-streaming → dbscan → agglomerative-clustering → feature-agglomeration → affinity-propagation → mean-shift → birch → bisecting-kmeans → optics → hdbscan → spectral-clustering → spectral-biclustering → spectral-coclustering |
| decomposition | pca → truncated-svd → incremental-pca → matrix-factorization-kernels → nmf → minibatch-nmf → fast-ica → factor-analysis → sparse-pca → minibatch-sparse-pca → dictionary-learning → minibatch-dictionary-learning → sparse-coder → latent-dirichlet-allocation → random-projection |
| model-selection | model-selection-core → grid-search-cv → halving-grid-search → parameter-grid → randomized-search → halving-random-search → leave-one-out-splitters → shuffle-splitters → group-shuffle-split → repeated-kfold → stratified-group-kfold → time-series-split → predefined-split → cross-validate-functions → learning-curve → validation-curve → permutation-test-score → threshold-tuning |
| pipeline | type-safe-pipeline → feature-union → pipeline-dag → streaming-pipeline → column-transformer → transformed-target-regressor |
| interop | arrow-interop → polars-interop → datafusion-interop → memory-mapped-interop → safetensors-internal-format → json-debug-format → model-export-retrofit → onnx-export → onnx-import → native-model-import → lora-adapters |
| gpu | gpu-opencl-backend → gpu-cuda-backend → gpu-rocm-backend |
| bindings | ffi-core → extract-language-bindings → lua-bindings → go-bindings |
| streaming | streaming-source-provider → sparse-streaming-batches → sparse-streaming-coverage |
| product | umbrella-crate → workflows-cli |
| python | python-bindings-complete → python-dataset-interop |
| julia | julia-dataset-interop |
| jvm | jvm-dataset-interop (deferred) |
| r | r-bindings |
| planning | sklearn-parity-matrix |
| contracts | access-pattern-contract → structured-tracing → mappable-fit-path → reopenable-fit-path → sample-weight-support → estimator-tags-and-callbacks → label-and-indicator-primitives → boxed-predictor-contract → multiclass-strategy → multioutput-strategy → resampling-primitives → frozen-estimator |
| datasets | dataset-generators → dataset-loaders |
| preprocessing | max-abs-scaler → normalizer → binarizer → function-transformer → kernel-centerer → label-encoder → label-binarizer → multi-label-binarizer → ordinal-encoder → k-bins-discretizer → power-transformer → quantile-transformer → spline-transformer → target-encoder → preprocessing-functions |
| feature-selection | variance-threshold → univariate-score-functions → univariate-selectors → select-from-model → rfe → sequential-feature-selector |
| naive-bayes | gaussian-naive-bayes → multinomial-naive-bayes → complement-naive-bayes → bernoulli-naive-bayes → categorical-naive-bayes |
| linear-model | sgd-classifier-regressor → penalized-glm-solver → regularization-path-solver → ridge-classifier → ridge-cv → lasso-cv → elastic-net-cv → logistic-regression-cv → multitask-linear → lars-family → orthogonal-matching-pursuit → bayesian-ridge → ard-regression → generalized-linear-regressors → quantile-regressor → huber-regressor → ransac-regressor → theil-sen-regressor → passive-aggressive → perceptron → sgd-one-class-svm |
| kernel | pairwise-kernel-functions → nusvm → linear-svr → kernel-ridge → kernel-approximation → gaussian-process → kernel-pca |
| neural-network | multilayer-perceptron → bernoulli-rbm |
| semi-supervised | label-propagation → label-spreading → self-training |
| text | text-tokenization-and-vocabulary → text-vectorizers → dict-vectorizer → feature-hasher |
| ensemble | extra-trees → adaboost → voting → stacking → random-trees-embedding |
| meta | dummy-estimators → isotonic-regression |
| calibration | calibrated-classifier |
| inspection | permutation-importance → partial-dependence |
| mixture | gaussian-mixture → bayesian-gaussian-mixture |
| discriminant | linear-discriminant-analysis → quadratic-discriminant-analysis |
| covariance | empirical-covariance → shrunk-covariance → ledoit-wolf → oas → min-cov-det → graphical-lasso → elliptic-envelope |
| cross-decomposition | pls-svd → pls-regression → pls-canonical → canonical-correlation-analysis |
| manifold | multidimensional-scaling → classical-mds → isomap → locally-linear-embedding → spectral-embedding → tsne |
| stats | streaming-moments → distribution-and-em-kernels → robust-statistics-kernels → binning-primitives |
| graph | graph-construction-and-algorithms |
| impute | iterative-imputer |
| image | image-patch-extraction |
<!-- dag-lanes:end -->

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
| W1 dataset-interop | Polars and DataFusion interop (Julia closes in W0; JVM stays deferred) | yes |
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
    if lane.in_progress is empty:
      head = first member of lane.merge_order not done/deferred
      if head.state == ready:
        launch one agent with head            # a blocked head blocks the whole lane
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
  `sklearn-parity-matrix`, `implementation-dag`) are `done` when their PRs merge and
  `mdbook build docs` is green; they carry `docs_only: true` where they list no cargo
  tests.
- Hardware-gated GPU tests may be skipped when no device is present; fallback tests must
  pass.

The cycle is executable: `scripts/dag-readiness.py` computes the states and the next
change per lane, `scripts/check-dag.py` is the consistency check, and
`scripts/check-completion.py` runs a change's completion checks. `runbook.md` holds the
exact command sequence.

## Risks / Trade-offs

- **Lane serialization reduces parallelism** → it prevents merge conflicts in shared
  crates; the parallel groups are still broad.
- **Join tail** → `model-export-retrofit`/`umbrella`/`python`/`cli` are late; split the
  retrofit by family and start the umbrella with core crates if needed.
- **Stale layers** → `dag.yaml` is the source of truth; the layer and lane blocks are
  generated and `scripts/check-dag.py` fails when they drift.
- **Completion drift** → checks are explicit; a change is not done until its PRs merge.

## Open Questions

None blocking. The readiness, consistency, completion, design-render and parity-matrix
scripts are implemented and covered by `scripts/tests/`.
