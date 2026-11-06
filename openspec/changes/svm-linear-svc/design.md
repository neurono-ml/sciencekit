# Design

## Context

See `proposal.md` — Why. `sciencekit_linear_model` already contains a LIBLINEAR-style
dual coordinate-descent solver for logistic regression, but its objective (logistic loss
vs squared hinge/hinge) and variable domain differ. `sciencekit_svm` will exist after
W3.5 and already owns the kernel/SMO modules; `sciencekit_math` provides the dense/sparse
dot products and the shared data view.

## Goals / Non-Goals

**Goals:**
- A fast, deterministic linear SVC for dense and sparse text-like data.
- One-vs-rest multiclass with clear decision semantics.
- Zero-copy sparse fit and prediction.
- A solver contract that mirrors LIBLINEAR's convergence behavior.

**Non-Goals:**
- Kernel functions, probability calibration, `Crammer-Singer` multiclass.
- Incremental/out-of-core fitting, `memmap2` and export.
- Reusing the logistic solver (different objective).

## Decisions

### Decision 1 — New dual coordinate-descent module in `sciencekit_svm`

`SKLinearSVC` lives in `sciencekit_svm` and implements the L2-regularized linear SVC dual:

```
min_α  1/2 αᵀ Q α - eᵀ α
s.t.   y ᵀ α = 0 is dropped in the bias-free formulation,
       0 ≤ α ≤ U, with U = C for L1-loss
       α ≥ 0 and Q ← Q + D for L2-loss, D = 1 / (2C)
```

and updates one dual coordinate at a time with the projected gradient, following Fan et
al. 2008 (LIBLINEAR Algorithm 3). The logistic solver is not reused: its loss, variable
domain and stopping rule are different, and sharing it would couple two crates.

Alternative considered: a generic QP solver (no dedicated QP API in `faer`; slower for
the sparse case) and the SMO solver from W3.5 (kernel-oriented, unnecessary overhead for
a linear model).

### Decision 2 — Dense and sparse share one coordinate loop

A small design trait exposes `row_dot(index, coefficients)` and `add_scaled_row(index,
scale, target)` over an owned dense `Array2` or a borrowed `CsMat`. The sparse path scans
only stored entries; no `n × p` dense copy is created. The solver keeps an `O(n)` dual
vector and an `O(p)` coefficient vector.

Alternative considered: converting CSR to CSC (helps column-wise scans but adds a copy
and does not remove the need for row dots in the dual update).

### Decision 3 — One-vs-rest multiclass, parallel heads

For `c` classes the classifier fits `c` binary heads (class `i` against the rest), each
with its own class-weighted `C`, and predicts the label with the largest decision value.
Ties break by the smallest canonical class index. Heads are independent and run in
parallel with rayon; the model stores one coefficient vector and intercept per head.

Alternative considered: `Crammer-Singer` (a different multiclass formulation, out of
scope) and one-vs-one (more models, no advantage for the linear case).

### Decision 4 — Class weights scale the effective regularization

`Balanced` computes `n_samples / (n_classes · class_count)` and explicit weights are used
as supplied. The weight multiplies the class's effective `C`, matching scikit-learn's
class-weighted linear SVC. The intercept is unpenalized and recovered from the dual
solution.

### Decision 5 — Convergence and determinism

The solver starts from zero dual coefficients, sweeps coordinates in a deterministic
order, and stops when the maximum projected gradient falls below `tolerance` or the
iteration budget is exhausted. Exhaustion returns `NotConverged { iterations }`. No
randomness is used, so repeated fits are identical.

### Decision 6 — Prediction surface

The model exposes `decision_function` (one value per class) and `predict_labels`
(argmax with canonical tie-breaking). `predict_probabilities` returns
`UnsupportedOperation` with guidance to a calibrated classifier; the linear SVC has no
probability model and fabricating one would be misleading. Prediction accepts dense or
sparse views and dispatches once.

### Decision 7 — Validation and observability

Builder defaults: `C = 1.0`, `SquaredHinge`, `class_weight = None`, `include_intercept =
true`, `tolerance = 1e-4`, `maximum_number_of_iterations = 1000`, `backend = Faer`,
`execution_mode = Automatic`. Fit transfers backend/execution mode to the model and
records the backend in every span. Errors: `EmptyInput`, `InvalidHyperparameter`,
`ShapeMismatch`, `NonFiniteValue`, `NotConverged`.

### Decision 8 — File layout

```
crates/sciencekit_svm/src/linear_svc/
  mod.rs
  builder.rs
  config.rs
  design.rs
  solver.rs
  core_implementation.rs
  model.rs
  fit.rs
  predict.rs
  *_tests.rs
  acceptance_tests/mod.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Dual CD convergence depends on `C` and data scaling** → documented tolerance and
  iteration budget; `NotConverged` is explicit.
- **Sparse row dots can be slower than dense for high density** → the design chooses the
  path from the representation; benchmark in the acceptance task.
- **Class-weighted C changes the effective objective** → documented and tested against
  the unweighted fit.
- **No probabilities** → `UnsupportedOperation`, documented; users can compose with a
  calibrated classifier.
- **One-vs-rest can be sensitive to class imbalance** → `Balanced` weights are the
  documented remedy.

## Migration Plan

Additive: a new estimator in `sciencekit_svm`. No existing API changes.

## Open Questions

None blocking. Probability calibration and `Crammer-Singer` are separate future changes.
