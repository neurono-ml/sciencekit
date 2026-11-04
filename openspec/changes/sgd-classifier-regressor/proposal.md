## Why

Wave 2 closes with W2.5. The four linear models already shipped solve their objective by
*touching the whole design*: `SKLinearRegression` and `SKRidge` form a Gram matrix or a
decomposition, and `SKLasso`/`SKElasticNet`/`SKLogisticRegression` iterate over a
**retained design** that grows with the stream. None of them can learn from data that never
fits in memory, and none of them exposes a per-sample learning loop.

`SKSGDClassifier` and `SKSGDRegressor` are the library's **first truly streaming supervised
estimators**: each update consumes one sample, so `fit_streaming` walks the source once and
discards every batch — peak memory is one batch plus `O(n_features · n_classes)`
coefficients, never the dataset. They also complete the scikit-learn parity surface the PRD
asks for (§12, Phase 2: *"`SKSGDClassifier`/`SKSGDRegressor` with `SKLazySource`"*) and close
the linear-model family before Wave 3 moves to neighbors and SVMs. The academic anchors are
already fixed by `wave-plan-foundation`: Bottou 2010 (stochastic gradient) and Pegasos
(Shalev-Shwartz et al. 2007).

## What Changes

- **New `sgd` substrate in `crates/sciencekit_linear_model`** — the shared learning loop:
  the sealed loss-function trait with the nine scikit-learn losses (five classifier, four
  regressor), the four learning-rate schedules (`Constant`, `Optimal`, `InverseScaling`,
  `Adaptive`), the proximal L1/L2/ElasticNet penalty update, per-sample class weights, the
  optional averaged iterate (ASGD), a self-contained seeded shuffler, per-epoch convergence
  and early-stopping bookkeeping, and one epoch driver over dense views, zero-copy CSR rows
  and streaming batches.
- **`SKSGDClassifier<F>` / `SKSGDClassifierModel<F>`** — one-vs-rest heads (scikit-learn's
  SGD is OvR, not multinomial), losses `hinge`, `log_loss`, `modified_huber`,
  `squared_hinge`, `perceptron`, `predict_labels` plus `predict_probabilities` restricted to
  the two probabilistic losses, `decision_function`, integer and nominal targets through
  `SKLabelTable`, and `class_weight`.
- **`SKSGDRegressor<F>` / `SKSGDRegressorModel<F>`** — losses `squared_error`, `huber`,
  `epsilon_insensitive`, `squared_epsilon_insensitive`, `SKRegressorPredictor<F>`,
  `epsilon` for the insensitive losses.
- **Builder surface (both):** `loss`, `penalty` (`L2`/`L1`/`ElasticNet`/`None`), `alpha`,
  `l1_ratio`, `include_intercept`, `learning_rate` schedule, `initial_learning_rate`,
  `inverse_scaling_power`, `tolerance`, `maximum_number_of_iterations`,
  `number_of_epochs_without_improvement`, `early_stopping`, `validation_fraction`,
  `shuffle`, `random_seed`, `average`, `warm_start_from`, `convergence_policy`
  (`Strict` by default, or `Permissive`), `backend(SKBackendKind)` (default `Faer`) and
  `execution_mode(SKExecutionMode)` (default `Automatic`); the classifier adds
  `class_weight`. Direct constructors stay private.
- **True incremental and out-of-core learning:** `fit` runs epochs over the in-memory design
  (shuffling with the seeded generator); `fit_streaming` makes **one pass** over a
  `SKSupervisedLazySource` without retaining the design; `partial_fit` on the fitted model
  continues the learning-rate clock and the average from where the fit stopped, so a stream
  can be replayed batch by batch.
- **Dense and zero-copy CSR:** `fit` consumes `SKDataView::Sparse` by iterating borrowed CSR
  rows, never densifying.
- **New structured error variant** `SKError::UnsupportedOperation { operation, reason }` for
  operations a valid configuration does not support (probabilities under a non-probabilistic
  loss, early stopping over a one-shot stream), added to the central taxonomy.
- **Shared class weighting:** `SKClassWeight` and its per-sample expansion move out of
  `logistic/config.rs` into a crate-shared module, re-exported through `logistic` so the
  existing logistic API is unchanged; both linear-model families use it.
- **Companion `*_tests.rs` modules** covering hand-checkable losses, each schedule, each
  penalty, the averaged iterate, dense/sparse/streaming agreement under a fixed seed,
  `partial_fit` continuation, label round-tripping, class weights, determinism, `f32`/`f64`,
  concurrency and the model boundary.

## Capabilities

### New Capabilities

- `stochastic-gradient-solver`: the shared substrate — loss-function contract and catalogue,
  learning-rate schedules, proximal penalty update, per-sample weighting, iterate averaging,
  seeded shuffling, convergence and early-stopping bookkeeping, and the dense/CSR/batch epoch
  driver.
- `stochastic-gradient-classifier`: the `SKSGDClassifier` estimator/model contract — builder
  surface and defaults, one-vs-rest heads, loss semantics and the probabilistic subset,
  target canonicalization, prediction surface, class weighting, warm start and the error
  taxonomy.
- `stochastic-gradient-regressor`: the `SKSGDRegressor` estimator/model contract — builder
  surface and defaults, loss semantics including the insensitive band, prediction and the
  error taxonomy.

### Modified Capabilities

- `error-model`: the central enum gains `UnsupportedOperation { operation, reason }`, so an
  operation unsupported by an otherwise valid configuration is reported as a named variant
  instead of an overloaded conversion or representation error.

## Impact

- **New code:** `crates/sciencekit_linear_model/src/sgd/` (substrate:
  `loss/`, `learning_rate.rs`, `penalty.rs`, `shuffle.rs`, `averaging.rs`, `epoch/`,
  `driver/` + companion `*_tests.rs`), `.../sgd_classifier/` and `.../sgd_regressor/`
  (`builder.rs`, `core_implementation.rs`, `model/`, `streaming/` + `*_tests.rs`).
- **Modified code:** `sciencekit_common/src/errors/error_kind.rs` (one variant); the
  `SKClassWeight` move with re-exports; `crates/sciencekit_linear_model/src/lib.rs` exports.
- **Dependencies:** none new — the shuffler is a small seeded generator inside the
  substrate; `ndarray`, `sprs`, `rayon` and `tracing` already cover the rest.
- **Public API:** additive except the internal `SKClassWeight` relocation, which keeps the
  `logistic::SKClassWeight` path resolving through a re-export. New: `SKSGDClassifier`, `SKSGDClassifierBuilder`,
  `SKSGDClassifierModel`, `SKSGDRegressor`, `SKSGDRegressorBuilder`, `SKSGDRegressorModel`,
  `SKSGDClassifierLoss`, `SKSGDRegressorLoss`, `SKSGDLearningRate`, `SKSGDPenalty`,
  `SKConvergencePolicy`, and the two warm-start coefficient types. The fitted models add
  `converged()` and `epochs_taken()` accessors.
- **Downstream:** the epoch driver and the proximal update are reused by `SKLinearSVC`
  (Wave 3, dual/primal coordinate descent) and by `SKMiniBatchKMeans`/`SKPCA` streaming later;
  the loss catalogue is the base for Wave 4's boosting gradients. `docs/src/algorithms.md`
  gains both entries and a new `Development Explained` chapter documents the family.
- **Acceptance (PRD §8.7/§10.3):** correct on small (hand-checkable) and large data; dense,
  sparse, streaming and incremental fits agree within the documented tolerance under a fixed
  seed; probabilities are normalized and available only for probabilistic losses; the
  estimator and model are `Send + Sync` and serve concurrent predictions; the model carries
  its backend kind and is the export boundary; model export and the metrics suite stay with
  their Phase 7 / W6.3 changes, as the previous linear-model changes already did.
