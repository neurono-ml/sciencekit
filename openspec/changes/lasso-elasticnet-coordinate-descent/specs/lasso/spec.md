## Purpose

Defines `SKLasso`, the L1-regularized linear regressor: a builder-configured,
backend-carrying estimator with a distinct fitted model whose objective adds an L1 penalty
`α‖β‖₁`, reachable from dense and sparse inputs and from a dense `fit`, an incremental
`partial_fit` and an out-of-core `fit_streaming`, with one sparse solution per configured
`alpha`.

## ADDED Requirements

### Requirement: Builder-only construction with explicit defaults

`SKLasso<F>` SHALL be constructed only through `SKLasso::<F>::build()`, which SHALL expose
the chainable configuration sink: `alpha` (a scalar `F` or a sequence of `F`, default `1.0`),
`include_intercept(bool)` (default `true`), `tolerance(F)` (default `1e-4`),
`maximum_number_of_iterations(usize)` (default `1000`), `positive(bool)` (default `false`),
`selection(SKSelection)` (default `Cyclic`), `screening(bool)` (default `true`),
`backend(SKBackendKind)` (default `Faer`) and `execution_mode(SKExecutionMode)` (default
`Automatic`); the direct constructor SHALL stay private. Because the configuration carries
the scalar `alpha`, the estimator MAY be generic over `F`.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it fits with a single `alpha` equal to one, includes an intercept, is unconstrained
  in sign, uses cyclic selection and screening, carries `SKBackendKind::Faer` and an
  `Automatic` execution intent

#### Scenario: Configuration flows through the model

- **WHEN** an estimator is built with `include_intercept(false)`, `positive(true)`, a
  non-default backend and a non-default tolerance
- **THEN** the fitted model reports no intercept, reports the positive constraint, carries
  that backend kind and honors the tolerance

### Requirement: The L1 penalty yields a sparse solution and the KKT conditions hold

The estimator SHALL minimize `(1/(2·n))‖y − Xβ‖₂² + α‖β‖₁`, producing **exactly zero**
coefficients for features the penalty excludes, and the converged coefficients SHALL satisfy
the Lasso KKT conditions: `x_jᵀr/n = α·sign(β_j)` for every active coordinate and
`|x_jᵀr/n| ≤ α` for every coordinate at zero.

#### Scenario: Irrelevant features are driven exactly to zero

- **WHEN** a design includes feature columns uncorrelated with the target under a sufficient
  penalty
- **THEN** those coefficients are exactly zero

#### Scenario: KKT conditions hold at the solution

- **WHEN** a fit has converged
- **THEN** the active coordinates' signed gradients equal `α` in magnitude and the inactive
  coordinates' gradients are at or below `α`, all within tolerance

#### Scenario: A zero penalty recovers ordinary least squares

- **WHEN** a full-rank design is fitted with `alpha` zero
- **THEN** the coefficients equal the ordinary least-squares solution within tolerance

#### Scenario: A large penalty zeroes every coefficient

- **WHEN** the penalty exceeds the largest gradient magnitude at the origin
- **THEN** every coefficient is zero and the intercept equals the target mean

### Requirement: Prediction returns responses in the model scalar

The fitted model SHALL implement regressor prediction, computing each response as the dot
product of the feature row with the selected coefficients plus the intercept, and return an
owned `Array1<F>`. Plain prediction SHALL be defined only for the single-`alpha`
configuration; with several alphas it SHALL return a structured error directing the consumer
to `predict_with_alpha_index`. The estimator SHALL NOT expose prediction.

#### Scenario: Predictions on new rows

- **WHEN** a single-`alpha` model predicts a new feature row
- **THEN** the response equals the fitted line at that row within tolerance

#### Scenario: Ambiguous prediction is refused

- **WHEN** plain prediction is called on a model fitted with three alphas
- **THEN** it returns a structured error and the consumer can predict with an explicit alpha
  index instead

### Requirement: One solution per alpha

A fit SHALL produce one coefficient vector and one intercept per configured `alpha`; the
fitted model SHALL expose `alphas()`, `coefficients()` shaped `n_alphas × n_features` and
`intercepts()` of length `n_alphas`, in the order the alphas were configured.

#### Scenario: A sequence yields one solution per value

- **WHEN** an estimator is fitted with `alpha` equal to a sequence of three values
- **THEN** `coefficients()` has three rows, `intercepts()` three entries, and growing the
  penalty shrinks every coefficient's magnitude

### Requirement: Sparse input stays sparse

The estimator SHALL accept a zero-copy `CsMatView` design without densifying it, and a sparse
fit's coefficients SHALL agree with the equivalent dense fit within the documented relative
tolerance.

#### Scenario: Sparse and dense agree

- **WHEN** the same design is fitted densely and as an equivalent CSR matrix
- **THEN** the coefficient vectors and intercepts agree within the documented relative
  tolerance

### Requirement: Incremental and streaming fits match the dense fit

`partial_fit` on the fitted model and `fit_streaming` over a lazy source SHALL retain the
design per batch and re-solve, converging to the dense coefficients and intercepts within
floating-point tolerance, regardless of batch order or size. An in-memory execution intent for
`fit_streaming` SHALL instead collect the batches and reuse the dense path.

#### Scenario: Split batches equal the whole

- **WHEN** a dataset is fitted at once and, separately, its batches are accumulated with
  `partial_fit`
- **THEN** the two models' coefficients and intercepts agree within tolerance

#### Scenario: Empty source is rejected

- **WHEN** a streaming source emits no batches
- **THEN** the fit returns a structured error and no model

### Requirement: Structured errors on invalid input

The estimator SHALL reject an empty design, a target count disagreeing with the row count,
any negative `alpha`, an empty alpha sequence, a non-positive `tolerance` and a zero iteration
budget, each with the central `SKError` taxonomy, and prediction SHALL reject a column count
differing from the fitted design.

#### Scenario: Empty input

- **WHEN** `fit` receives zero rows
- **THEN** it returns a structured error before touching the solver

#### Scenario: Negative penalty

- **WHEN** an estimator is built with a negative `alpha` and fitted
- **THEN** the fit returns an invalid-hyperparameter error naming `alpha`

#### Scenario: Prediction shape mismatch

- **WHEN** the model predicts on a matrix whose column count differs from the fitted design
- **THEN** it returns a shape-mismatch error

### Requirement: Backend and execution intent are carried and recorded

Per the `backend-selection` capability, every estimator and model operation SHALL record the
configured backend kind in its observability span; `fit` SHALL transfer the backend kind into
the fitted model, and the kind SHALL never be hardcoded. The coordinate-descent iteration
dispatches no concrete backend, which the model records honestly.

#### Scenario: Configured kind reaches the model

- **WHEN** an estimator is built with a non-default backend and fitted
- **THEN** the model reports that backend kind and its span records the kind

### Requirement: The fitted model is shareable across threads

The fitted model SHALL be `Send + Sync` by construction so a single model serves concurrent
predictions without external synchronization.

#### Scenario: One model, many threads

- **WHEN** the same fitted model predicts on distinct inputs from multiple threads
- **THEN** every prediction matches the single-threaded result
