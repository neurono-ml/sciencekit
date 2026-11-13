# Proposal

## Why

Wave 3 closes with the SVM family: the library already has linear models, stochastic
gradient and the evaluation metrics from `metrics-foundation`, but no kernel method. Two
gaps block it: `sciencekit_math` has pairwise *distances* but no *kernel functions*, and
no crate implements a sequential-minimal-optimization (SMO) solver. `SKSVC` and `SKSVR`
are the canonical non-linear baselines and the reference against which `SKLinearSVC`
(W3.6) will be validated.

## What Changes

- `sciencekit_math` gains a `kernels` module (new capability `kernel-functions`) with
  linear, polynomial, radial-basis-function, sigmoid and precomputed kernels. Every kernel
  accepts dense, sparse and mixed inputs, dispatches once per operation and never
  densifies a sparse input.
- New sub-crate `sciencekit_svm` with `SKSVC<F>` and `SKSVR<F>`.
- `SKSVC`: C-SVC formulation, one-vs-one multiclass, class weights, the kernel catalogue
  above, `C`, `tolerance`, `maximum_number_of_iterations`, `cache_size`, shrinking, and
  the mandatory builder surface (`backend` default `Faer`, `execution_mode` default
  `Automatic`).
- `SKSVR`: epsilon-insensitive support-vector regression with `C`, `epsilon`, the same
  kernel catalogue and the same builder surface.
- A pure-Rust SMO solver: second-order working-set selection, KKT stopping criterion,
  optional shrinking, a bounded in-memory kernel cache and deterministic iteration for a
  fixed configuration.
- Distinct fitted models carrying support vectors, dual coefficients, the bias term, the
  kernel configuration and the class table. `SKSVC` implements
  `SKClassifierPredictor` (`predict_labels`, `predict_probabilities`) plus
  `decision_function`; `SKSVR` implements `SKRegressorPredictor`.
- Calibrated probabilities: Platt scaling for the binary decision value and pairwise
  coupling for the one-vs-one multiclass case.
- Dense, sparse and mixed feature inputs throughout fit and prediction; sparse data is
  never densified.
- Validation errors for non-finite input, empty input, invalid `C`/`epsilon`/kernel
  parameters, and `NotConverged` when the iteration budget is exhausted.
- Explicit non-goals recorded in the PRD: `memmap2`-backed kernel caches, persistence and
  model export are Wave 7 (interop) work.
- Companion `*_tests.rs` modules covering known small fixtures, large data, kernel
  equivalence across representations, determinism, concurrency and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `kernel-functions`: the dense/sparse/mixed kernel catalogue used by SVM (and later by
  kernel PCA and kernel ridge), with its mathematical definitions, representation parity
  and dispatch contract.
- `support-vector-machines`: `SKSVC` and `SKSVR` — builder surface, SMO solver contract,
  multiclass one-vs-one behavior, probability calibration, support-vector model state,
  representation parity, validation and convergence semantics.

### Modified Capabilities

None.

## Impact

- **Code:** new `crates/sciencekit_svm` workspace member and a new
  `crates/sciencekit_math/src/kernels/` submodule; root `Cargo.toml` membership.
- **APIs:** additive. New kernel functions, `SKSVC`, `SKSVR`, their builders and fitted
  models.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `ndarray`, `sprs`, `rayon`;
  no new external crate. `sciencekit_metrics` is a dev-dependency for acceptance tests.
- **Downstream:** unblocks `SKLinearSVC` (W3.6), kernel PCA / kernel ridge in later waves,
  and gives the pipeline a non-linear classifier.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metric production through `sciencekit_metrics`, companion `*_tests.rs` modules. Model
  export is deferred to Wave 7 and that deferral is recorded in the PRD by the associated
  documentation change.
