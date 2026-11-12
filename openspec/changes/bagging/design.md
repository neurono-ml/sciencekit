# Design

## Context

See `proposal.md` — Why. The estimator contracts are `SKSupervisedFit` with associated
`Model`/`Error`, and `SKClassifierPredictor`/`SKRegressorPredictor` on the model. There
is no `SKEstimator` trait: the code separates fit by supervision axis. `sciencekit_ensemble`
will already own the bootstrap sampler from `random-forest`; this change reuses it.

## Goals / Non-Goals

**Goals:**
- A compositional bagging estimator that works with any compatible base estimator.
- Feature-subset routing that is correct for both fit and predict.
- Deterministic, parallel and OOB-capable fitting.

**Non-Goals:**
- Heterogeneous base estimators or a runtime trait-object registry.
- Stacking/voting and streaming/`partial_fit`.
- `memmap2` backing and export.

## Decisions

### Decision 1 — Two generic structs, not one trait

`SKBaggingClassifier<B>` and `SKBaggingRegressor<B>` are structs generic over the
configured base estimator `B`. The classifier requires `B: SKSupervisedFit<F>` and
`B::Model: SKClassifierPredictor<F>`; the regressor requires the analogous predictor.
This follows the existing supervision-axis split and keeps the type system honest: a
regressor configuration cannot be passed to the classifier.

Alternative considered: a single `SKBagging` with a runtime `SKEstimator` trait object
(rejected — the fit method is generic over `X`/`T`, so the trait cannot be object-safe,
and dynamic dispatch would lose zero-copy generics).

### Decision 2 — No monolithic `SKEstimator` supertrait

A marker supertrait would only repeat `Model`/`Error`; it would not make bagging generic
over the fit method because `SKSupervisedFit::fit<X, T>` is generic in its inputs. The
bounds are written directly on the bagging structs. A `SKEstimator` marker can be added
later for pipelines (Wave 6) if it earns its place; it is not required here.

### Decision 3 — Error composition

The base fit returns `B::Error: From<SKError>`. Bagging defines
`SKBaggingError<E> { Central(SKError), Base(E) }` implementing `From<SKError>` (producing
`Central`), `SKBaggingError::from_base` / `From<SKBaseEstimatorError<E>>` (producing `Base`
without colliding with `From<SKError>` when `E = SKError`), and `From<SKBaggingError<SKError>>
for SKError` (for integration with central scorers and pipelines), and uses it as its
`fit`/`predict` error type. This preserves the base estimator's specific error instead of
flattening it into the central enum.

Alternative considered: requiring `B::Error = SKError` (too restrictive) and
`B::Error: Into<SKError>` (not guaranteed by the trait and loses algorithm-specific
detail).

### Decision 4 — Bootstrap rows and features

A per-estimator seed `splitmix64(global_seed ^ i)` drives the row sample and, when
`bootstrap_features` is enabled, the feature subset. The feature subset is an
`Array1<usize>`/`Vec<usize>` stored on the model. Fit materializes the row sample in the
input representation (dense `Array2` or CSR row copy, never densifying) and then takes
the feature subset, producing a borrowed view for the base estimator.

### Decision 5 — Feature-subset routing at prediction

For estimator `i`, prediction builds a view over the stored feature indices and calls the
base model. The full feature matrix is never leaked to an estimator that did not see those
columns. This is the main correctness trap of bagging and is covered by a dedicated test.

### Decision 6 — Aggregation

Classifier: average the base probabilities over the canonical class table and predict the
largest with the smallest-index tie-break. Regressor: average the base responses. Both
parallelize over rows and preserve order.

### Decision 7 — OOB estimation

The shared bootstrap mask from `random-forest` records which rows each estimator did not
see. The model scores only those rows, using accuracy for classification and R² for
regression through `sciencekit_metrics`. Rows with no OOB estimator are excluded; the
model reports how many rows were scored. OOB without bootstrap is rejected at build.

### Decision 8 — Representation behavior

Bagging passes the representation through to the base estimator. It does not advertise a
representation the base estimator does not support; the base estimator's
`UnsupportedRepresentation` is surfaced through `SKBaggingError::Base`. When the input is
sparse and the base supports CSR, the bootstrap sample stays CSR. Dense-only base
estimators receive dense input and reject sparse input themselves.

### Decision 9 — Determinism and parallelism

Estimators are fitted in parallel with independent slots; seeds depend only on the global
seed and the estimator index, so results are thread-count independent. Prediction is
row-parallel and deterministic.

### Decision 10 — File layout

```
crates/sciencekit_ensemble/src/bagging/
  mod.rs
  classifier.rs
  regressor.rs
  builder.rs
  config.rs
  model.rs
  fit.rs
  predict.rs
  error.rs
  *_tests.rs
  acceptance_tests/mod.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Generic base estimators increase compile-time monomorphization** → bounded by the
  number of base types used; acceptable for a library.
- **Feature-subset routing is easy to get wrong** → a dedicated test compares predictions
  with a manually subsetted estimator.
- **Error type nesting** → `SKBaggingError<E>` is explicit; callers can match the central
  or base variant.
- **Heterogeneous ensembles are unsupported** → documented non-goal; users can wrap a
  common enum themselves.
- **OOB coverage depends on sample size** → the model reports scored rows and errors when
  none exist.

## Migration Plan

Additive: new estimators in `sciencekit_ensemble`. No existing API changes.

## Open Questions

None blocking. A runtime heterogeneous ensemble and stacking are future changes.
