# Design

## Context

See `proposal.md` — Why. `sciencekit_outlier` already owns `SKIsolationForest`, the shared
`SKContamination` enum and the export helpers for a fitted detector.
`sciencekit_neighbors` exposes `SKKDTree`, `SKBallTree`, `SKBruteForceSearcher` and the
object-safe `SKNearestNeighborsSearcher<F>` contract with exact `k_nearest` and
`radius_neighbors`. `sciencekit_svm` exposes the SMO solver pieces (working-set selection,
gradients, step, shrinking, stopping and kernel cache) but its public solver entry points
are written for the C-SVC/SVR duals. `sciencekit_math::kernels` exposes
`SKKernelParameters<F>` and the linear, polynomial, radial-basis-function and sigmoid
kernels over dense, sparse and mixed views.

The reference behaviour is scikit-learn's `LocalOutlierFactor(novelty=...)` and
`OneClassSVM(kernel=..., nu=...)`. Both estimators return negative values for outlier
scores and implement `decision_function`/`predict`; the typed label surface belongs to
sciencekit and does not leak scikit-learn's `-1`/`1` integers.

## Goals / Non-Goals

**Goals:**
- Exact LOF geometry with the documented k-neighbour and reachability definitions.
- One-Class SVM with the four non-precomputed kernels and `nu` in `(0, 1]`.
- Builder-only configuration, `backend` in every span, typed labels and offsets.
- Dense/sparse parity without densification, through an explicit data-view decision.
- Export and import for both fitted models.

**Non-Goals:**
- Missing values, streaming/`partial_fit`, GPU, `memmap2`.
- Precomputed kernels for One-Class SVM.
- LoOP, incremental LOF, multi-class novelty and score normalization beyond the offset.

## Decisions

### Decision 1 — New folder modules inside `sciencekit_outlier`

The crate already exists and owns the outlier contract, `SKContamination` and the export
conventions, so the detectors live there rather than in a new crate:

```
crates/sciencekit_outlier/src/
  local_outlier_factor/{mod.rs, builder.rs, config.rs, geometry.rs, fit.rs,
                        model.rs, predict.rs, export.rs, *_tests.rs}
  one_class_svm/{mod.rs, builder.rs, config.rs, dual.rs, solver.rs, fit.rs,
                 model.rs, predict.rs, export.rs, *_tests.rs}
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines and tests
live in companion `*_tests.rs` modules declared at the end.

### Decision 2 — LOF builder surface and defaults

`SKLocalOutlierFactor::<F>::build()` returns `SKLocalOutlierFactorBuilder<F>`, mirroring
`SKIsolationForest`. The builder exposes:

| Setter | Default | Validation |
|---|---|---|
| `nearest_neighbors_count` | 20 | `1..=n_samples` at fit |
| `contamination` | `Auto` | `Auto` or a fraction in `(0, 0.5]` |
| `novelty` | `false` | none |
| `metric` | `Euclidean` | Euclidean, Manhattan or Chebyshev |
| `backend` | `Faer` | none |
| `execution_mode` | `Automatic` | none |

`SquaredEuclidean` is rejected: LOF takes a maximum over distances and divides by them, so
the squared variant would change the geometry. The error names the parameter and the three
accepted metrics.

Alternative considered: exposing the searcher kind (KD tree, ball tree, brute force) as a
public setter. Rejected — it leaks an implementation choice into the estimator surface and
the representation already determines the sound option.

### Decision 3 — LOF geometry

For a point `x` with its `k` nearest distinct training neighbours `N_k(x)`:

1. `core_distance(x) = d(x, n_k)`, the distance to the k-th neighbour.
2. `reachability_distance(x, y) = max(core_distance(y), d(x, y))`.
3. `local_reachability_density(x) = 1 / mean_{y ∈ N_k(x)} reachability_distance(x, y)`.
4. `local_outlier_factor(x) = mean_{y ∈ N_k(x)} lrd(y) / lrd(x)`.

A density of zero (all reachability distances zero, i.e. more than `k` duplicate points)
yields an LOF of `1` for the affected point and is documented and tested; it never
produces NaN or infinity. The model exposes `negative_outlier_factor = -lof` so larger
values mean more normal, matching `SKOutlierPredictor::score_samples`.

### Decision 4 — Novelty semantics

- `novelty = false` (default): the model stores the training neighbours and their local
  reachability densities, and exposes `fit_predict` and `negative_outlier_factor`. It has
  no out-of-sample geometry, so `predict`, `decision_function` and `score_samples` on new
  data return `SKError::UnsupportedOperation { operation: "predict", reason: ... }`
  suggesting `novelty(true)`. This is a typed refusal, not a panic.
- `novelty = true`: the model stores the training index and densities; predicting a new
  row finds its `k` training neighbours, computes the new row's local reachability density
  and the factor against the stored neighbour densities. `fit` does not assign labels to
  the training rows in this mode, matching scikit-learn's partition of the two workflows.

`contamination` always sets the offset from the quantile of the fit-time LOF scores in
both modes; the offset is exposed and shifts `score_samples` into `decision_function`.

### Decision 5 — Data-view boundary decision (LOF)

`data-view-boundary` requires an explicit decision. LOF accepts dense, sparse and mixed
inputs, so no dense-only exception is needed:

- **Dense fit**: build a `SKKDTree` for the axis-aligned metrics (Euclidean, Manhattan,
  Chebyshev), which is the cheapest exact index, and query `k_nearest` for every training
  row. A ball tree is the alternative if a future metric needs it; the decision is internal
  and does not change the API.
- **Sparse fit**: build `SKBruteForceSearcher::build(training, metric)`, which routes
  through the pairwise dispatcher and never densifies. The tree geometry is dense by
  construction, so the tree is deliberately not used for sparse training.
- **Novelty prediction**: dense query against a dense index uses the tree; any pair with a
  sparse participant uses the brute-force searcher over the stored training data. Dispatch
  happens once per operation, not per row.

This mirrors the pattern established by `dbscan` and `knn-classifier-regressor`, and means
`UnsupportedRepresentation` is never returned for LOF. The sparse branch costs
`O(n_query · n_train)` and is documented as such.

### Decision 6 — One-Class SVM formulation

The estimator solves the one-class dual with the shared SMO machinery:

- Kernel matrix over the training rows from `SKKernelParameters<F>`; the kernel is
  precomputed once per fit into the bounded cache that `sciencekit_svm` already provides.
- The dual is the standard `min ½ αᵀKα` subject to `0 ≤ α_i ≤ 1/(νn)` and `Σ α_i = 1`,
  solved with second-order working-set selection and the KKT stopping criterion.
- `rho = Σ α_i K(x_i, x_j)` for a free support vector, averaged over free support vectors
  to reduce round-off.
- `decision_function(x) = Σ α_i K(x_i, x) − rho`; `score_samples` is the same value and
  `predict` maps non-negative values to `Inlier` and negative values to `Outlier`. The
  offset exposed by `SKOutlierPredictor` is `rho`.

`nu` is validated in `(0, 1]`; the upper bound `1/(νn)` is validated after fit and a
`nu` so small that every row is bounded is still accepted (it degenerates to a hard-margin
solution, as in scikit-learn). Non-finite kernel output during fit returns
`NonFiniteValue`; an exhausted iteration budget returns `NotConverged`.

Alternative considered: writing a separate one-class solver. Rejected — the SMO working
set, cache, shrinking and stopping logic is identical and is already tested; the change
adds a one-class problem formulation and loop, not a second solver.

### Decision 7 — Kernel and representation parity (One-Class SVM)

The four kernels are selected through the existing `SKKernel` catalogue (`Linear`,
`Polynomial`, `RadialBasisFunction`, `Sigmoid`); `Precomputed` is rejected at build time
because the estimator never receives a precomputed training kernel. `gamma` follows the
catalogue's `Scale`/`Auto`/`Value` policy, and `degree`/`coef0` are validated exactly as in
`svm-svc-svr`. Dense, sparse and mixed inputs go through the kernel functions directly, so
sparse data is never densified and no representation error is produced.

### Decision 8 — Export and import (no refusal)

Both fitted models implement `SKExportable<F>` and `SKImportable<F>`:

- **LOF** exports the training design, the neighbour count, the metric, the contamination
  offset, the novelty flag, the stored local reachability densities and the
  `negative_outlier_factor` vector. The training design is exported as dense tensors or as
  CSR `values`/`indices`/`indptr` tensors with a `training_representation` metadata
  parameter, so a sparse fit round-trips without densification.
- **One-Class SVM** exports the support vectors (dense or CSR triplet), dual coefficients,
  `rho`, `nu`, kernel parameters and backend.

A typed refusal is not used because `SKIsolationForestModel` already implements the same
contract and the Wave 7 export retrofit expects every fitted model to export; refusing
would leave a hole in `export_coverage` with no technical justification.

### Decision 9 — Validation, observability and concurrency

- Builder validation: `nearest_neighbors_count > 0`, contamination in range, accepted
  metric, `nu` in `(0, 1]`, degree > 0, finite `coef0` and explicit `gamma`.
- Fit validation: non-empty input, at least `k + 1` distinct rows, finite values, feature
  count consistency.
- Prediction validation: feature-count mismatch and non-finite queries.
- Spans record the configured `SKBackendKind` on `fit`, `predict`, `decision_function`,
  `score_samples` and `fit_predict`; `fit` transfers the kind into the fitted model and the
  model records it on its own operations. Dispatch from kind to backend happens per
  operation, never in `build`.
- Every fitted model is `Send + Sync`; concurrent prediction tests assert equality with the
  sequential result.

## Risks / Trade-offs

- **LOF is sensitive to duplicate points and zero densities** → the zero-density rule is
  explicit (`LOF = 1`) and tested; no NaN can escape.
- **The sparse branch is O(n_query · n_train)** → documented; the dense tree branch is the
  fast path and the API never forces sparse users to densify.
- **Reusing the SMO machinery requires a one-class loop that is not C-SVC's** → the new
  loop is covered by hand-checkable two-point fixtures and by comparing against the
  C-SVC-style KKT conditions.
- **`novelty = false` cannot predict new data** → typed `UnsupportedOperation` with a
  suggestion, instead of a silent wrong answer.
- **Exporting the LOF training data makes states large** → inherent to out-of-sample LOF;
  documented, with CSR export to avoid densifying sparse fits.
- **`nu` near zero degenerates to a hard-margin solution** → accepted and documented, as in
  scikit-learn.

## Migration Plan

Additive: two estimators, two builders, exports and tests. No existing estimator changes.
The catalog row for LOF/One-Class SVM moves from "Planned" to implemented only when this
change lands.

## Open Questions

None blocking. LoOP, incremental LOF and missing-value routing are future changes.
