# Design

## Context

See `proposal.md` — *Why*. The design is constrained by what the repository already provides:

- `sciencekit_common` supplies the estimator/model split (`SKSupervisedFit<F>`, fitted-model
  prediction traits), `SKTargetView`, `SKLabelTable` / `sk_canonicalize_labels`,
  `SKBuilder` / `SKBuilderState`, `SKBackendKind`, `SKExecutionMode`, the central `SKError`,
  `sk_run_operation`, and the streaming contract (`SKDataBatch`,
  `SKSupervisedBatch`, `SKLazySource`, `SKSupervisedLazySource`,
  `sk_run_supervised_streaming_driver`).
- Streaming batches are **dense only**: `SKDataBatch` and `SKSupervisedBatch` own
  `Array2<F>`. Sparse out-of-core input is therefore not reachable through the current
  streaming contract; sparse support lives on the in-memory `fit`.
- The workspace has **no RNG dependency** (`ndarray`, `sprs`, `rayon`, `tracing` only), and
  `sciencekit_linear_model` is a pure-Rust crate with no LAPACK/BLAS requirement.
- `SKClassWeight<F>` and its per-sample expansion currently live inside
  `logistic/config.rs` and are exported as `logistic::SKClassWeight`.
- The library convention is `SKError::NotConverged { iterations }` when an iterative process
  exhausts its budget (ridge, lasso/elastic-net, logistic all do this), whereas scikit-learn
  emits a warning and returns the last iterate.
- `SKSGDClassifier`/`SKSGDRegressor` are named by PRD §12; `SGD` is an acronym in the same
  class of names as `SKKDTree`, `SKSVC`, `SKPCA`. Parameter names still follow the
  no-abbreviation rule (`maximum_number_of_iterations`, not `max_iter`).

## Goals / Non-Goals

**Goals:**

- One shared learning loop whose per-sample update serves dense, CSR and streaming inputs
  with identical semantics, so streaming is not a second implementation.
- A loss catalogue split by task family, with a single sealed trait the loop consumes.
- An estimator/model pair per task that is immutable, `Send + Sync`, backend-carrying and
  scikit-learn-compatible on the surface that matters (losses, penalties, schedules,
  averaging, early stopping, OvR).
- `partial_fit` that genuinely continues learning (schedule counter, average, class table),
  not a retained-design re-solve.

**Non-Goals:**

- Multinomial/softmax SGD (scikit-learn's `SGDClassifier` is one-vs-rest only).
- `sag`/`saga`/`liblinear` solvers, which W2.4 already owns.
- Sparse streaming, which needs an extension of the streaming batch contract; tracked for a
  later change.
- Parallelizing the per-sample loop: SGD's update is sequential by definition, and
  parallelizing it changes the trajectory rather than the answer.
- Model export and metrics (Phase 7 / W6.3), as with every Wave 1–2 change.

## Decisions

### 1. Module layout

`crates/sciencekit_linear_model/src/sgd/` is the substrate:

- `loss/` — `definition.rs` (the sealed `SKSGDLoss` trait), `classifier.rs`, `regressor.rs`,
  `mod.rs` (dispatcher + re-exports).
- `learning_rate/schedule.rs`, `learning_rate/optimal.rs`, `learning_rate/mod.rs`.
- `penalty.rs` (the `SKSGDPenalty` value and the proximal update), `class_weight.rs`
  (re-export of the shared policy), `shuffle.rs` (`SKRandomGenerator`, Fisher–Yates),
  `averaging.rs`, `config.rs` (validation), `update.rs` (the single-sample update),
  `objective.rs` (running epoch loss), `early_stopping.rs`.
- `epoch/` — `dense.rs`, `sparse.rs`, `batch.rs`, `mod.rs` (the three drivers over one
  update closure).
- `state.rs` — the mutable learning state (weights, intercept, counter, average, schedule
  rate), the only thing the drivers mutate.

`src/sgd_classifier/` and `src/sgd_regressor/` mirror the existing estimator layout
(`builder.rs`, `core_implementation.rs`, `model/`, `streaming/`, `test_datasets.rs` +
companion `*_tests.rs`). Every `mod.rs` is a pure dispatcher; every file stays under 200
lines.

### 2. Loss contract: sealed trait + one public enum per family

`SKSGDLoss<F>` exposes `value(prediction, target)`, `slope(prediction, target)` (the
subgradient with respect to the linear predictor), `is_probabilistic()` and
`positive_probability(prediction)`. It is sealed to the crate. `SKSGDClassifierLoss` and
`SKSGDRegressorLoss` are the public, exhaustively matched enums; each delegates to a
zero-sized implementation. Rationale: a single enum would let a regressor loss reach the
classifier path and force runtime family checks in the hot loop. Alternative considered: one
`SKSGDLoss` enum with a `family()` method — rejected for the same misconfiguration risk.

Probabilities: `log_loss` uses the logistic sigmoid; `modified_huber` uses the clipped
affine `(prediction + 1) / 2` in `[0, 1]`, matching scikit-learn. Other losses report
`is_probabilistic() == false` and the model rejects the request.

### 3. The per-sample update

For sample `(x, y)` with weight `c`, the update is

```
gradient  = c · slope(wᵀx + b, y) · x
w        ← w − η · (gradient + α · (1 − l1_ratio) · w)      (elastic-net L2 part)
b        ← b − η · gradient_intercept                        (never penalized)
w        ← soft_threshold(w, η · α · l1_ratio)               (L1/elastic-net)
```

`L2` is the limit `l1_ratio = 0`, `L1` the limit `l1_ratio = 1` with no L2 gradient, `None`
removes both. The public selection is `SKSGDPenalty { L2, L1, ElasticNet, None }`, default
`L2`. The data term is the *mean* loss (like scikit-learn), so `α` is not divided by the
sample count; this keeps `alpha` semantics identical to `SGDClassifier`. The intercept
receives the same learning rate and no penalty. Alternative: a per-sample objective with an
explicit proximal operator computed through a subgradient — rejected as algebraically
equivalent but slower.

### 4. Learning-rate schedules

- `Constant`: `η_t = η₀`.
- `InverseScaling`: `η_t = η₀ / (1 + t)^p`.
- `Optimal`: `η_t = 1 / (α · (t + t₀))`, with `t₀ = max(1, 1 / (α · s))` and `s` the mean
  absolute slope over the first batch (a cheap estimate of the initial loss scale). This is
  scikit-learn's asymptotic form; the constant is estimated rather than copied, so it is not
  bit-identical — documented in the spec's "deterministic and selectable" wording.
- `Adaptive`: `η` is held at its current value and halved after a non-improving epoch.
  Because `Adaptive` needs epoch feedback it is only meaningful for `fit`; over a
  one-shot stream it degenerates to `Constant`, which the spec allows.

`Optimal` requires `α > 0`; with `α = 0` the build rejects the combination with
`InvalidHyperparameter` rather than dividing by zero.

### 5. Averaging

With `average(true)`, after each update the state keeps `w̄ ← w̄ + (w − w̄) / t` where `t`
counts updates since averaging began. The averaged vector (and intercept) is what the model
reports and predicts with; the last iterate is not retained for prediction. Rationale: ASGD
reduces the variance of the final iterate at `O(p)` extra memory, which fits the library's
"large and small data" acceptance criterion. The counter and the average live in the model
state so `partial_fit` continues them.

### 6. Shuffling and the random generator

A `SplitMix64` generator plus Fisher–Yates (partial) shuffle lives in `sgd/shuffle.rs`, seeded
by `random_seed` (default `0`) — reproducible by default, a deliberate deviation from
scikit-learn's `random_state=None`. The generator produces **row indices**, so the same seed
yields the same order for a dense matrix, a CSR matrix and a batch; this is what makes the
streaming/in-memory agreement test meaningful. Alternatives: adding `rand`/`rand_xoshiro`
(rejected: a new dependency for ~40 lines, plus MSRV and build-time cost); using
`ndarray::Array::random` (not available without `rand`).

### 7. Convergence, budget and early stopping

The epoch driver accumulates the loss as it iterates, yielding an epoch objective. A fit
converges when the epoch objective improves by no more than `tolerance` for
`number_of_epochs_without_improvement` consecutive epochs.

The convergence **policy** decides what an unconverged fit does. `Strict` (default) returns
`SKError::NotConverged` when the full requested budget executes without converging, matching
ridge/lasso/logistic. `Permissive` returns the last (or averaged) iterate, records
`converged() == false` and emits a `tracing::warn!`. The policy exists because an SGD budget
is a compute knob, not a feasibility test: discarding a long streamed fit because the budget
ran out is hostile, and scikit-learn returns the model with a warning.

A **provider-limited** fit is never a budget exhaustion. When a one-shot source supplies
fewer passes than the budget (e.g. one pass out of a thousand), the fit returns the model
with `converged() == false` and a warning even under `Strict`; only a fit that actually
executed the whole requested budget can raise. The model exposes `epochs_taken()` so the
distinction is observable without logs.

`early_stopping(true)` reserves a deterministic prefix-decimated validation slice of
`validation_fraction` (never shuffled together with training), scores it once per epoch and
keeps the best iterate. Because a one-shot source has no epochs to compare, `fit_streaming`
with early stopping returns `SKError::UnsupportedOperation` — an explicit refusal rather than
a silent no-op.

### 8. Streaming: one pass, by construction

`fit_streaming` walks a `SKSupervisedLazySource` once. It cannot rewind, and the existing
`fit_streaming(&mut S)` convention used by every other estimator is not re-openable, so
multi-epoch learning over a stream is expressed as repeated `partial_fit` calls or by
re-creating the source. `maximum_number_of_iterations` therefore governs `fit` only; the
streaming path runs one epoch, reports `epochs_taken() == 1`, `converged() == false` and a
warning. Peak memory is one batch plus `O(n_features · n_classes)` state, which the spec
asserts. The follow-up change `streaming-source-provider` replaces this restriction with a
reopenable provider that honors the epoch budget; this change does not depend on it.

### 9. Dense, sparse and batch drivers

The three drivers share one `update` closure and differ only in how they yield a row: a
dense `ArrayView1`, a borrowed `CsMatView` outer row, or a batch row. The sparse driver
iterates `outer_iterator()` and never converts to dense. The learning state is the only
mutable object; the drivers take it by `&mut`. `fit` chooses the driver from
`SKDataView::{Dense, Sparse}`.

### 10. Classifier heads and prediction

One-vs-rest: a binary problem yields one head (`+1` label and its complement); `k > 2`
classes yield `k` heads, each fitted on `y_i == class` versus the rest with `±1` targets.
`predict_labels` is the arg-max of the heads' linear scores; ties resolve to the lowest
canonical index. `predict_probabilities` is only defined for the probabilistic losses,
normalized per row; otherwise `UnsupportedOperation`. `decision_function` returns the
`n_samples × n_heads` matrix. Alternative: multinomial softmax — rejected for scikit-learn
parity and because the per-sample update would need a full softmax per step.

### 11. Shared class weighting

`SKClassWeight<F>` and `sk_expand_class_weights` move to a new
`src/class_weight/` module. `logistic` re-exports the type (`logistic::SKClassWeight` keeps
resolving), and the SGD modules re-export it under `sgd::SKClassWeight`. Alternative:
duplicating the enum — rejected; the balanced formula and its validation have to stay in one
place. The logistic spec's requirement is unchanged because its behavior does not change.

### 12. New error variant

`SKError::UnsupportedOperation { operation: &'static str, reason: String }` is added to the
`#[non_exhaustive]` enum. Two behaviors need it: probabilities for a non-probabilistic loss,
and early stopping over a one-shot stream. Reusing `UnsupportedRepresentation` would
misreport the cause; `InvalidHyperparameter` would misplace the failure at build time.
Alternative: a crate-local `SKSGDError` — rejected because the central taxonomy already
carries the same class of failure for every algorithm and the project forbids ad-hoc
error types in central contracts.

### 13. Warm start is an explicit coefficient source

`warm_start_from` takes `SKSGDClassifierCoefficients<F>` or
`SKSGDRegressorCoefficients<F>` — the model's class table (classifier) plus coefficients and
intercepts — and seeds the learning state, leaving the estimator immutable and `Send + Sync`.
This follows W2.4's explicit-source warm start instead of scikit-learn's mutable instance
state. A source whose feature count or class table disagrees is rejected with a structured
error.

### 14. Backend, observability and concurrency

Every public operation runs in `sk_run_operation` with the configured `SKBackendKind`
(default `Faer`), even though the loop performs no heavy algebra: the `backend-selection`
spec requires a uniform selection surface. `fit` transfers the kind into the model; the
model's `predict` records it. The estimator and model hold no interior mutability and are
`Send + Sync`. The learning loop is sequential; `rayon` is not used inside it, and the
docs explain why (parallel updates would change the trajectory, not compute the same
answer faster).

## Risks / Trade-offs

- **The convergence criterion is noisy** (online epoch loss) → early stopping uses the
  held-out slice; the plateau rule is the fallback; tests use fixed seeds and closed-form
  datasets.
- **`Optimal` is not bit-identical to scikit-learn** → documented; tests assert the
  asymptotic form and convergence, not equality with sklearn.
- **One streaming pass may underfit** → documented; `partial_fit` continues learning, and
  the acceptance test compares one pass to one in-memory epoch, which is the honest claim.
- **L1 plus averaging can lag the active set** → the averaged vector is reported only after
  the update count is large enough; tests compare against the non-averaged fit within a loose
  tolerance and assert exact zeros only without averaging.
- **The budget is a hard error under `Strict`, unlike scikit-learn's warning** → `Permissive`
  returns the iterate with `converged() == false`; the default stays consistent with the
  library's `NotConverged` convention, and a provider-limited stream never raises.
- **A new error variant touches `sciencekit_common`** → additive to a `#[non_exhaustive]`
  enum, so no downstream match becomes non-exhaustive; the delta spec records it.
- **Randomness can make tests flaky** → every stochastic test fixes `random_seed`; tests
  that assert closed forms run with `shuffle(false)`.

## Migration Plan

- Additive change: new modules and exports, one enum variant, one internal relocation with
  re-exports. No migration for existing users. The `SKClassWeight` path stays valid.
- Rollback is deleting the new modules and reverting the relocation; no data or deployment
  state is involved.

## Open Questions

- Whether the seeded generator should move to `sciencekit_common` when Wave 4 (random forest,
  bagging) needs it — deferrable; the module boundary makes the move mechanical.
- Whether `fit_streaming` should accept a source factory for multi-epoch streaming — a
  cross-crate signature decision for a later change, not needed for this one.
