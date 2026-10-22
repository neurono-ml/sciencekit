## Purpose

Defines `SKElasticNet`, the mixed L1/L2-regularized linear regressor: a builder-configured,
backend-carrying estimator with a distinct fitted model whose objective adds
`α·ρ‖β‖₁ + (α·(1−ρ)/2)‖β‖₂²`, reachable from dense and sparse inputs and from a dense `fit`,
an incremental `partial_fit` and an out-of-core `fit_streaming`, with one solution per
configured `alpha` and a `l1_ratio` that interpolates between Lasso and Ridge.

## ADDED Requirements

### Requirement: Builder-only construction with explicit defaults

`SKElasticNet<F>` SHALL be constructed only through `SKElasticNet::<F>::build()`, which SHALL
expose the chainable configuration sink: `alpha` (a scalar `F` or a sequence of `F`, default
`1.0`), `l1_ratio(F)` (default `0.5`), `include_intercept(bool)` (default `true`),
`tolerance(F)` (default `1e-4`), `maximum_number_of_iterations(usize)` (default `1000`),
`positive(bool)` (default `false`), `selection(SKSelection)` (default `Cyclic`),
`screening(bool)` (default `true`), `backend(SKBackendKind)` (default `Faer`) and
`execution_mode(SKExecutionMode)` (default `Automatic`); the direct constructor SHALL stay
private. `l1_ratio` SHALL be validated to the closed interval `[0, 1]`.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it fits with a single `alpha` equal to one, `l1_ratio` equal to one half,
  includes an intercept, carries `SKBackendKind::Faer` and an `Automatic` execution intent

#### Scenario: Out-of-range l1_ratio is rejected

- **WHEN** `l1_ratio` is negative or greater than one and `build` is called
- **THEN** it returns an invalid-hyperparameter error naming `l1_ratio`

### Requirement: The l1_ratio interpolates between Lasso and Ridge

The estimator SHALL minimize
`(1/(2·n))‖y − Xβ‖₂² + α·ρ‖β‖₁ + (α·(1−ρ)/2)‖β‖₂²`, with `ρ = l1_ratio`. At `ρ = 1` the
solution SHALL equal the Lasso solution for the same `alpha`; at `ρ = 0` it SHALL equal the
pure-L2 closed form; the converged coefficients SHALL satisfy the ElasticNet KKT conditions:
`x_jᵀr/n = α·(1−ρ)·β_j + α·ρ·sign(β_j)` for active coordinates and `|x_jᵀr/n| ≤ α·ρ` for
coordinates at zero.

#### Scenario: l1_ratio one reproduces Lasso

- **WHEN** a design is fitted by `SKElasticNet` with `l1_ratio` one and by `SKLasso` with the
  same `alpha`
- **THEN** the two coefficient vectors agree within tolerance

#### Scenario: l1_ratio zero reproduces the pure-L2 closed form

- **WHEN** an orthogonal design is fitted with `l1_ratio` zero
- **THEN** each coefficient equals `(xⱼᵀy)/(‖xⱼ‖₂² + α·n)` within tolerance

#### Scenario: KKT conditions hold at the solution

- **WHEN** a fit has converged with `l1_ratio` strictly between zero and one
- **THEN** the active coordinates' gradients match their L1/L2 subgradients and the inactive
  coordinates' gradients are at or below the L1 threshold, all within tolerance

### Requirement: Prediction and the multi-alpha model shape

The fitted model SHALL implement regressor prediction as `X·β + β₀` for the selected `alpha`
and SHALL expose `alphas()`, `l1_ratio()`, `coefficients()` shaped `n_alphas × n_features`
and `intercepts()` of length `n_alphas`, in configuration order. Plain prediction SHALL be
defined only for the single-`alpha` configuration; with several alphas it SHALL return a
structured error directing the consumer to `predict_with_alpha_index`.

#### Scenario: Predictions on new rows

- **WHEN** a single-`alpha` model predicts a new feature row
- **THEN** the response equals the fitted line at that row within tolerance

#### Scenario: Ambiguous prediction is refused

- **WHEN** plain prediction is called on a model fitted with several alphas
- **THEN** it returns a structured error and the consumer can predict with an explicit alpha
  index instead

### Requirement: Sparse input, incremental and streaming fits

The estimator SHALL accept a zero-copy `CsMatView` design without densifying it, and a sparse
fit SHALL agree with the equivalent dense fit within the documented relative tolerance.
`partial_fit` and `fit_streaming` SHALL retain the design per batch and re-solve, converging
to the dense fit within tolerance regardless of batch order or size, and an empty source SHALL
be rejected with a structured error.

#### Scenario: Sparse and dense agree

- **WHEN** the same design is fitted densely and as an equivalent CSR matrix
- **THEN** the coefficient vectors and intercepts agree within the documented relative
  tolerance

#### Scenario: Split batches equal the whole

- **WHEN** a dataset is fitted at once and, separately, its batches are accumulated with
  `partial_fit`
- **THEN** the two models' coefficients and intercepts agree within tolerance

### Requirement: Structured errors on invalid input

The estimator SHALL reject an empty design, a target count disagreeing with the row count,
any negative `alpha`, an empty alpha sequence, a non-positive `tolerance`, a zero iteration
budget and an out-of-range `l1_ratio`, each with the central `SKError` taxonomy, and
prediction SHALL reject a column count differing from the fitted design.

#### Scenario: Empty input

- **WHEN** `fit` receives zero rows
- **THEN** it returns a structured error before touching the solver

#### Scenario: Negative penalty

- **WHEN** an estimator is built with a negative `alpha` and fitted
- **THEN** the fit returns an invalid-hyperparameter error naming `alpha`

### Requirement: Backend and execution intent are carried and recorded

Per the `backend-selection` capability, every estimator and model operation SHALL record the
configured backend kind in its observability span; `fit` SHALL transfer the backend kind into
the fitted model, and the kind SHALL never be hardcoded.

#### Scenario: Configured kind reaches the model

- **WHEN** an estimator is built with a non-default backend and fitted
- **THEN** the model reports that backend kind and its span records the kind

### Requirement: The fitted model is shareable across threads

The fitted model SHALL be `Send + Sync` by construction so a single model serves concurrent
predictions without external synchronization.

#### Scenario: One model, many threads

- **WHEN** the same fitted model predicts on distinct inputs from multiple threads
- **THEN** every prediction matches the single-threaded result
