# Design

## Context

See `proposal.md` — Why. `model-selection-core` supplies `SKIndexSplit`, the splitter
contract and row gathering. `metrics-ranking` and `metrics-clustering` supply the scoring
surface. `sciencekit_common` supplies the estimator contracts and builder state. The crate
is `sciencekit_model_selection`.

## Goals / Non-Goals

**Goals:**
- Deterministic, parallel candidate/fold evaluation.
- A type-safe candidate list instead of stringly-typed parameter maps.
- Classifier and regressor wrappers with the matching prediction surface.

**Non-Goals:**
- Halving/randomized search (W6.5).
- Nested CV, stacking, custom scorer objects beyond the documented policies.
- `memmap2` and export.

## Decisions

### Decision 1 — Candidates are configured estimator values

A candidate is a fully configured `E` (for example a `SKLogisticRegression` or `SKRidge`).
The user builds the cartesian product and passes `Vec<E>`. This is Rust-idiomatic and keeps
the grid type-safe; a string-keyed parameter map would require runtime downcasting.
`E: Clone` so folds can fit independently.

### Decision 2 — Separate classifier and regressor grid searches

`SKGridSearchCVClassifier<E>` requires `E: SKSupervisedFit<F>` and
`E::Model: SKClassifierPredictor<F>`; the regressor analog requires
`SKRegressorPredictor<F>`. This avoids specialization and gives each wrapper the correct
prediction surface. The scoring policy is a closed enum per family:
classification (accuracy, F1, precision, recall, AUC, log loss, balanced accuracy) and
regression (negative MSE, negative MAE, R²).

### Decision 3 — Fold evaluation and aggregation

For each candidate and fold, gather the train/test blocks, fit a fresh model, score it and
store the score in a `(candidates × folds)` matrix. Candidates and folds are independent;
the evaluation runs in parallel and results are written by index. Mean and standard
deviation are computed per candidate; the winner is the highest mean, ties by candidate
index.

### Decision 4 — Refit

When `refit` is true, the winning candidate is fitted on the full data and stored as the
best model. The wrapper's `predict` delegates to it. When false, the model has no best
model and prediction returns `UnsupportedOperation`, mirroring the reference semantics.

### Decision 5 — Error score

`SKErrorScore` is either `Raise` (propagate the first candidate error) or `Value(F)`
(record the value for the failing fold and continue). A candidate that fails every fold
gets that value as its mean; it loses only when the value is worse than every achievable score; `NaN` is guaranteed to rank below any finite mean, and any other value is the caller's declared penalty.

### Decision 6 — Representation and grouping

Fold gathering uses `sk_gather_rows`, so dense and sparse inputs stay in their
representation. Group vectors are passed to the splitter; a splitter that needs groups
validates their length once.

### Decision 7 — Observability and validation

The grid search records a `fit` span and per-fold spans with the candidate index; the
candidate estimators record their own backend. Builder validation rejects empty candidates
and scorer/estimator family mismatches. Errors: `InvalidHyperparameter`, `ShapeMismatch`,
`UnsupportedOperation` and the candidate/scorer errors.

### Decision 8 — File layout

```
crates/sciencekit_model_selection/src/grid_search/
  mod.rs, configuration.rs, results.rs, error_score.rs, observation.rs, prediction.rs
  builder/{mod.rs, classifier_builder.rs, regressor_builder.rs, state.rs, *_tests.rs}
  scoring/{mod.rs, classifier.rs, regressor.rs, *_tests.rs}
  evaluation/{mod.rs, candidate_evaluation.rs, fold_scorer.rs, *_tests.rs}
  classifier/{mod.rs, estimator.rs, fit.rs, model.rs, predict.rs, *_tests.rs}
  regressor/{mod.rs, estimator.rs, fit.rs, model.rs, predict.rs, *_tests.rs}
  mock_estimators/
  concurrency_tests.rs, representation_tests.rs
```

`mod.rs` files are pure dispatchers; companion `*_tests.rs` modules are declared at the
end of their dispatcher.
