# Design

## Context

See `proposal.md` — *Why* for motivation. The design is constrained by what already exists in
the repository:

- `sciencekit_common` already provides the classifier contracts this change consumes:
  `SKClassifierPredictor<F>` (`predict_labels -> Array1<i64>`,
  `predict_probabilities -> Array2<F>`), `SKSupervisedFit<F>`, `SKTargetView` (`Integer`,
  `Nominal`, `Continuous`), `SKLabelTable` / `sk_canonicalize_labels`, `SKBuilder` /
  `SKBuilderState`, `SKBackendKind`, `SKExecutionMode`, `SKError`, the supervised streaming
  contract (`SKSupervisedBatch`, `SKSupervisedLazySource`,
  `sk_run_supervised_streaming_driver`) and `SKLabelScorer`.
- `sciencekit_math` provides the backend abstraction (`SKBackendSelection` /
  `sk_math_backend_for_kind`) and the higher-order kernels; the logistic solvers are
  reductions over the design and reuse `azip!`/`par_azip!` and `rayon`.
- `sciencekit_linear_model` already established the estimator/model split, the private
  constructor + builder convention, the per-operation backend dispatch, the `*_tests.rs`
  companion-module convention, the folder-module rule (a `mod.rs` is a pure dispatcher, no
  file over 200 lines) and — from W2.3 — the **retained-design accumulator** pattern and the
  shared dense/sparse column-access idea that the logistic solvers generalize.
- The MSRV is Rust 1.85 / edition 2024. No new dependency is required; the solvers are pure
  Rust and do not require LAPACK/BLAS.

## Goals / Non-Goals

**Goals:**

- A single `SKLogisticRegression` estimator covering the full scikit-learn penalty × solver ×
  multi-class matrix, with a pure-Rust default path.
- A shared `logistic` solver substrate whose design abstraction serves dense and CSR inputs
  without densifying, and whose accumulator serves the incremental and streaming regimes.
- A fitted model that is a distinct, backend-carrying export boundary implementing
  `SKClassifierPredictor<F>` and exposing a decision function and the class table.

**Non-Goals:**

- `newton-cg` / `newton-cholesky` / `sag` solvers and the `liblinear` `none` penalty: out of
  scope; the portfolio is `{Lbfgs, Liblinear, Saga}` (proposal).
- Model export (Safetensors/JSON) and the metrics suite (Phase 7 / W6.3).
- `SGDClassifier`/`SGDRegressor` (W2.5) and the linear SVM (W3), which reuse the SAGA
  substrate but are separate changes.
- Instance-state `warm_start` (rejected in the ADR; the estimator stays immutable).

## Decisions

### 1. Module layout (folder modules, dispatcher `mod.rs`)

`crates/sciencekit_linear_model/src/logistic_regression/` holds the estimator:
`builder.rs`, `core_implementation.rs`, `model/` (`core_implementation`, `predictor`,
`partial_fit`, `mod`), `streaming/` (`batches`, `fit`, `state`, `mod`) and companion
`*_tests.rs`. `crates/sciencekit_linear_model/src/logistic/` holds the substrate: `config.rs`
(+ validation), `resolution.rs` (penalty/solver/multi-class), `class_weight.rs`,
`design.rs` (the dense/sparse abstraction), `objective.rs` (loss + gradient + proximal step),
`accumulator.rs`, and `solver/` (`lbfgs`, `saga`, `liblinear`, `convergence`, `mod`). Every
`mod.rs` only declares submodules and re-exports; every implementation file stays under 200
lines, splitting into folders as it grows.

### 2. Objective scaling and the C/alpha duality

Internally the substrate stores a single regularization strength. To mirror scikit-learn we
fix the objective as `(1/(2·C))·R(w) + (1/n)·Σ_i c_i·loss_i`, i.e. the penalty is divided by
`C` (equivalently multiplied by `alpha = 1/C`) and the data term is the **mean** loss. The
builder's `alpha(F)` sets the strength directly (`alpha = value`), `C(F)` sets it as
`alpha = 1/value`; the last setter wins and a `tracing::warn!` is emitted when both are
called. Rationale: keeps the public surface faithful (`C`) *and* coherent with
`SKRidge`/`SKLasso` (`alpha`), while the solvers see one number. Alternative considered: a
single `RegularizationStrength` enum — rejected as heavier for no observable gain.

### 3. Solver portfolio

- **L-BFGS-B + orthant-wise (OWL-QN) step** (`solver/lbfgs.rs`): a limited-memory BFGS with a
  bounded curvature history for the smooth `L2`/`None` objective, and a pseudo-gradient /
  orthant-projection step so the same iteration can also handle an L1 component. This is the
  default and the multinomial workhorse.
- **SAGA** (`solver/saga.rs`): a variance-reduced incremental-gradient method with a
  per-sample gradient table and a proximal soft-threshold step for the L1 component. It is
  the natural path for `L1`/`ElasticNet` and for sparse designs; its per-sample iteration maps
  cleanly onto a borrowed CSR row.
- **LIBLINEAR dual coordinate descent** (`solver/liblinear.rs`): a shrinking dual
  coordinate-descent iteration for `L1`/`L2` one-vs-rest, matching scikit-learn's default for
  small binary problems and fastest on very sparse, high-dimensional designs.

Alternatives considered: (a) a single generic gradient solver — rejected, it cannot handle
the non-smooth L1 term and is poor on ill-conditioned sparse designs; (b) adding
`newton-cg`/`newton-cholesky` — rejected as out of the wave plan's scope and requiring dense
Hessians that fight the sparse goal.

All solvers share `solver/convergence.rs`: convergence is checked against `tolerance` on the
appropriate criterion (projected-gradient norm for L-BFGS-B, the duality gap / gradient
criterion for SAGA and the dual solver), bounded by `maximum_number_of_iterations`, and
exhausting the budget raises `SKError::NotConverged { iterations }` — never a stale vector.

### 4. Multi-class resolution

`multi_class(Auto)` resolves once, before the solve: multinomial for `{Lbfgs, Saga}` when
`n_classes > 2`, one-vs-rest for `Liblinear` and for binary problems. `OvR` and `Multinomial`
force the strategy; `Liblinear` + `Multinomial` is rejected in `resolution.rs` with a
structured error. Binary problems always fit a single coefficient vector (the OvR model with
one head). Multinomial fits `n_classes` heads jointly under the softmax; OvR fits
`n_classes` independent binary heads and normalizes their sigmoids at prediction.

### 5. Targets, labels and prediction

`fit` destructures `SKTargetView`: `Integer` and `Nominal` canonicalize through
`SKLabelTable` (nominal via `sk_canonicalize_labels`), `Continuous` is rejected. The fitted
model stores the class table plus a **binary flag**. `predict_labels` returns the arg-max of
the decision function (equivalently the probabilities); `predict_probabilities` applies
sigmoid (binary), softmax (multinomial) or normalized OvR sigmoids; `decision_function`
returns `X·Wᵀ + b`. The estimator has no prediction methods — only the model does
(`SKClassifierPredictor<F>` lives on the model, mirroring the regressor split).

### 6. Shared design abstraction and the retained-design accumulator

`solver/design.rs` exposes one trait over a dense `ArrayView2<F>` and a borrowed
`CsMatView<F>` giving per-row dot products, gradient accumulation and CSR row iteration;
solvers never densify a sparse design, retaining only the `O(n · n_classes)` linear predictor.
`accumulator.rs` generalizes W2.3's retained-design accumulator to also carry canonical
labels and per-sample weights, and is the single backing store for `partial_fit` and
`fit_streaming`; the objective is not a function of the Gram matrix, so the design is kept and
re-solved at read time (documented memory trade-off). An in-memory execution intent for
`fit_streaming` collects the batches and reuses the dense path.

### 7. Class weights and warm start

`class_weight` normalizes to per-sample weights `c_i` (unweighted penalty): `None → 1`,
`Balanced → n_samples / (n_classes · class_count)`, explicit → indexed by canonical class
order; negatives and all-zero weights are rejected. `warm_start(true)` seeds the optimizer
from coefficients supplied on the builder (a `SKLogisticCoefficients` value, typically a
previous model's), keeping the estimator immutable and `Send + Sync`; without a source it
starts from zero, matching the cold start. This is the documented deviation from
scikit-learn's instance-state reuse.

### 8. Observability and backend

Every public operation runs inside an `sk_run_operation` span recording the configured
`SKBackendKind` (never hardcoded), `fit` transfers the kind into the model, and the model's
own operations record it. The solvers dispatch no concrete backend (they are reductions over
the design), which the model records honestly, as W2.3 does for coordinate descent.

### 9. Concurrency

The estimator and the fitted model are immutable and `Send + Sync`; the multinomial/OvR heads
are solved over `rayon` where independent, and a single fitted model serves concurrent
predictions. `thread_tests.rs` asserts one model, many threads.

## Risks / Trade-offs

- **Three solvers × two multi-class strategies is a large surface** → factor the shared
  objective, convergence and design abstraction first, and TDD each solver against a
  hand-checkable closed form (a separable design, an orthogonal design, a single-feature
  binary problem) before wiring the estimator.
- **L-BFGS-B/OWL-QN and SAGA numerics differ across penalties** → assert cross-solver
  agreement on shared datasets (dual solver vs. L-BFGS under L2; SAGA vs. L-BFGS under L2)
  within tolerance, and KKT/optimality checks.
- **Sparse designs must not densify** → the design abstraction is the only path solvers use,
  and a test asserts no dense `n × p` copy for a CSR fit (as W2.3 does).
- **Retained design grows with the stream** → documented memory trade-off, identical to
  W2.3's L1 accumulator; the incremental regime is opt-in via `partial_fit`/`fit_streaming`.
- **Probability calibration depends on the multi-class strategy** → normalization tests
  (rows sum to one, ordering matches the decision function) plus a deterministic seed for
  SAGA so tests are reproducible.
- **`C`/`alpha` dual surface can surprise** → last-setter precedence plus an explicit warning;
  covered by dedicated tests.

## Open Questions

- None blocking. The exact default `maximum_number_of_iterations` (100, matching
  scikit-learn) and the SAGA step-size schedule are tuning details confined to the substrate
  and can be adjusted without changing the specs.
