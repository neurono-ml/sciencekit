## Why

Every out-of-core fit in the library consumes a one-shot `SKLazySource` /
`SKSupervisedLazySource` by `&mut`, so it can read the data **once**. That is the right
default for exact accumulators (a Gram matrix, a running mean, a category table), but it
makes the stochastic estimators — `SKSGDClassifier`/`SKSGDRegressor` first — unable to take
the number of passes their optimizers need: a single pass over a stream is one epoch, and
one epoch of SGD usually underfits. The wave plan promised `SKSGD*` "with `SKLazySource`",
not "with exactly one `SKLazySource`".

The obvious reach — accept "something convertible into a source" — does not work in Rust:
`Into<Source>` cannot infer its target type from a generic parameter, `impl<T> From<T> for T`
already claims every concrete source, and a closure factory has no nameable type for a user
to implement a conversion for. The library needs a **source provider seam** that both a
concrete source and a factory satisfy, plus one shared rule for how many passes an estimator
may take. Doing it once — for every estimator, with tests — replaces twelve near-identical
`fit_streaming` signatures with one contract.

## What Changes

- **New provider contract** in `sciencekit_common`: `SKSourceProvider<F>` (feature-only) and
  `SKSupervisedSourceProvider<F>` (features plus targets), each opening one pass at a time
  and declaring whether it can be reopened.
- **Two implementations per family**: the residual `&mut S where S: SKLazySource` (one pass,
  no wrapper, existing call sites unchanged) and `SKReopenableSource<G>` wrapping a
  `FnMut() -> Result<S, E>` factory (one fresh source per pass).
- **One epoch rule** in the streaming executor: an iterative estimator runs up to
  `maximum_number_of_iterations` passes; a one-shot provider is consumed exactly once and the
  executor warns through `tracing` when a larger budget was requested; a non-iterative
  (accumulating) estimator always takes exactly one pass because re-feeding would double
  count its statistics.
- **All eleven `fit_streaming` implementations migrate** to the provider seam:
  `SKStandardScaler`, `SKMinMaxScaler`, `SKRobustScaler`, `SKPolynomialFeatures`,
  `SKSimpleImputer`, `SKLinearRegression`, `SKRidge`, `SKLasso`, `SKElasticNet`,
  `SKLogisticRegression` and the new `SKSGDClassifier`/`SKSGDRegressor`.
- **`SKSGDClassifier`/`SKSGDRegressor` become multi-epoch over streams**: with a reopenable
  provider, `fit_streaming` runs the same epoch budget as `fit`, so a streamed fit can match
  the in-memory fit instead of approximating it with one pass. `SKOneHotEncoder`'s
  iterator-based `fit_streaming` remains a documented exception in this change: its batches
  are categorical views, not numeric blocks, and unifying them needs a categorical source
  contract of their own.
- **Tests**: every migrated estimator gains a reopenable-source test asserting
  `fit_streaming` equals `fit` on the concatenated data; the SGD estimators gain multi-epoch
  stream-versus-memory equivalence; a shared harness proves that a one-shot source is still
  consumed exactly once and that a non-iterative estimator rejects a second pass.
- **Documentation**: a `Development Explained` chapter on source providers and epochs, plus
  updates to the streaming chapter of the book.

## Capabilities

### New Capabilities

- `streaming-source-provider`: the provider seam — the feature-only and supervised traits,
  the one-shot and reopenable implementations, the epoch rule per estimator class, the
  budget/warning policy and the errors for a request the provider cannot satisfy.

### Modified Capabilities

- `streaming-batches`: the estimator-facing streaming contract changes from "consume a
  sequential source" to "consume a source provider and take one or more passes according to
  the estimator's class", while one-shot sources keep working through the residual
  implementation.

## Impact

- **New code:** `sciencekit_common/src/batching/provider/` (`mod`, `source_provider`,
  `supervised_provider`, `reopenable`, `mod_tests` + companion `*_tests.rs`) and the
  executor entry point that loops over passes.
- **Modified code:** every `fit_streaming` signature and its tests across
  `sciencekit_preprocessing`, `sciencekit_impute` and `sciencekit_linear_model`; the SGD
  streaming driver gains the epoch loop.
- **Dependencies:** none new; the GAT-based trait needs only stable Rust (MSRV 1.85).
- **Public API:** additive traits and one wrapper type; the existing
  `fit_streaming(&mut source)` call shape keeps compiling because `&mut S` implements the
  provider. Call sites that want multiple passes pass `SKReopenableSource::new(factory)`.
- **Sequencing:** lands after `sgd-classifier-regressor`. If that change is still open, its
  streaming requirement is amended in place; if it has archived, this change carries the
  matching delta.
- **Acceptance (PRD §8.7/§10.3):** every migrated estimator's streamed fit equals its dense
  fit on the concatenated data; a one-shot source is consumed exactly once; a non-iterative
  estimator rejects a second pass; a reopenable source over the same data reproduces the
  in-memory result; concurrency and `f32`/`f64` are covered.
