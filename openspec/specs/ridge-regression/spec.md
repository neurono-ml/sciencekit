# ridge-regression Specification

## Purpose
Defines `SKRidge`, the regularized linear regressor: a builder-configured,
backend-carrying estimator with a distinct fitted model, whose objective adds an L2 penalty,
reachable from dense and sparse inputs and from a dense `fit`, an incremental `partial_fit`
and an out-of-core `fit_streaming`, with one solution per configured `alpha`.

## Requirements

### Requirement: Builder-only construction with explicit defaults

`SKRidge<F>` SHALL be constructed only through `SKRidge::<F>::build()`, which SHALL expose the
chainable configuration sink: `alpha` (a scalar `F` or a sequence of `F`, default `1.0`),
`include_intercept(bool)`, `solver(SKRidgeSolver)`, `tolerance(F)`,
`maximum_number_of_iterations(usize)`, `backend(SKBackendKind)` and
`execution_mode(SKExecutionMode)`; the direct constructor SHALL stay private. Because the
configuration carries the scalar `alpha`, the estimator MAY be generic over `F`.

#### Scenario: Default configuration

- **WHEN** an estimator is built without calling any setter
- **THEN** it fits with a single `alpha` equal to one, includes an intercept, uses the
  `Automatic` solver, carries `SKBackendKind::Faer` and an `Automatic` execution intent

#### Scenario: Iterative-solver knobs are settable

- **WHEN** an estimator is built with an explicit `tolerance` and
  `maximum_number_of_iterations`
- **THEN** the fitted model's sparse solve honors those values

#### Scenario: Configuration flows through the model

- **WHEN** an estimator is built with `include_intercept(false)`, a non-default solver and a
  non-default backend
- **THEN** the fitted model reports no intercept, records that solver and carries that backend
  kind

### Requirement: The solver portfolio resolves deterministically

The builder SHALL expose a solver selection of `Automatic`, `Cholesky`, `Svd` and `SparseCg`.
`Automatic` SHALL resolve to `Cholesky` for dense input and to `SparseCg` for sparse input;
`Cholesky` SHALL fall back to `Svd` when `alpha` is zero or the Cholesky factor fails. The
solver actually used SHALL be exposed by the fitted model.

#### Scenario: Automatic chooses per representation

- **WHEN** the same data is fitted under `Automatic`, once dense and once as an equivalent
  sparse matrix
- **THEN** the dense fit records `Cholesky` and the sparse fit records `SparseCg`

#### Scenario: Cholesky falls back to SVD

- **WHEN** a rank-deficient design is fitted under `Automatic` with `alpha` zero
- **THEN** the fit succeeds, records `Svd` as the solver used and surfaces the rank

### Requirement: The Cholesky solver solves the centered normal equations

The `Cholesky` solver SHALL form the centered cross products `Ĉ = XᵀX − n·x̄x̄ᵀ` and
`ĉ = Xᵀy − n·x̄·ȳ`, add `alpha` to the diagonal and solve `(Ĉ + αI)β = ĉ` through the
backend's Cholesky, consuming dense and sparse input without densifying the design matrix.

#### Scenario: Exact fit on a hand-checkable system

- **WHEN** a small full-rank dense system is fitted under `Cholesky` with a known `alpha`
- **THEN** the coefficients equal the closed-form ridge solution within tolerance

#### Scenario: Sparse input stays sparse

- **WHEN** a sparse row-major design is fitted under `Cholesky`
- **THEN** the fit succeeds and densifies nothing beyond the `p × p` Gram and the result

### Requirement: The SVD solver is the robust dense path

The `Svd` solver SHALL decompose the centered design once and apply each `alpha` as
`β = V diag(σ/(σ²+α)) Uᵀ ỹ`, so a rank-deficient design with `alpha` zero still yields the
minimum-norm solution; it SHALL reject sparse input with a structured representation error
pointing at `Cholesky` or `SparseCg`.

#### Scenario: Rank deficiency with zero penalty

- **WHEN** a design with a duplicated column is fitted under `Svd` with `alpha` zero
- **THEN** the fit succeeds and the reported rank is below the design width

#### Scenario: Sparse input is rejected by the SVD solver

- **WHEN** a sparse design is fitted under `Svd`
- **THEN** the fit returns an unsupported-representation error and no model

### Requirement: The sparse CG solver never forms the Gram

The `SparseCg` solver SHALL minimize `(DᵀD + αP)β = Dᵀy` by conjugate gradient, applying
`DᵀD` through a zero-copy operator over the borrowed sparse matrix so `XᵀX` is never
materialized, honoring the configured `tolerance` and `maximum_number_of_iterations`, and
SHALL raise a structured non-convergence error when the iteration budget is exhausted.

#### Scenario: Sparse fit matches the dense reference

- **WHEN** a sparse design is fitted under `SparseCg` and densely under `Cholesky`
- **THEN** the two coefficient vectors agree within the documented relative tolerance

#### Scenario: Exhausting the iteration budget

- **WHEN** `SparseCg` runs with `maximum_number_of_iterations` set to zero
- **THEN** the fit returns a non-convergence error carrying the iteration count and no model

### Requirement: The intercept is never penalized

Every solver SHALL leave the intercept out of the penalty: the Gram and SVD solvers SHALL
center the data and recover `β₀ = ȳ − x̄ᵀβ`, and the CG solver SHALL use the algebraically
equivalent unpenalized augmentation. With `include_intercept(false)` the model SHALL report a
zero intercept and solve through the origin.

#### Scenario: Penalty does not shrink the intercept

- **WHEN** data whose best line has a large intercept is fitted with growing `alpha`
- **THEN** the intercept stays near its unpenalized estimate while the slopes shrink

#### Scenario: Intercept disabled

- **WHEN** the same data is fitted with `include_intercept(false)`
- **THEN** the model reports a zero intercept

### Requirement: One solution per alpha

A fit SHALL produce one coefficient vector and one intercept per configured `alpha`; the
fitted model SHALL expose `alphas()`, `coefficients()` shaped `n_alphas × n_features` and
`intercepts()` of length `n_alphas`, in the order the alphas were configured.

#### Scenario: Scalar alpha yields one solution

- **WHEN** an estimator with a single scalar `alpha` is fitted
- **THEN** `coefficients()` has one row and `intercepts()` has one entry

#### Scenario: A sequence yields one solution per value

- **WHEN** an estimator is fitted with `alpha` equal to a sequence of three values
- **THEN** `coefficients()` has three rows, `intercepts()` three entries, and the row order
  matches the configured order

### Requirement: Prediction returns responses in the model scalar

The fitted model SHALL implement regressor prediction, computing each response as the dot
product of the feature row with the selected coefficients plus the intercept, and return an
owned `Array1<F>`. Plain prediction SHALL be defined only for the single-`alpha`
configuration; when several alphas were configured it SHALL return a structured error
directing the consumer to `predict_with_alpha_index`. The estimator SHALL NOT expose
prediction.

#### Scenario: Predictions on new rows

- **WHEN** a single-`alpha` model predicts a new feature row
- **THEN** the response equals the ridge-fitted line at that row within tolerance

#### Scenario: Ambiguous prediction is refused

- **WHEN** plain prediction is called on a model fitted with three alphas
- **THEN** it returns a structured error and the consumer can predict with an explicit alpha
  index instead

### Requirement: Incremental and streaming fits match the dense fit

`partial_fit` on the fitted model and `fit_streaming` over a lazy source SHALL accumulate the
same centered cross-product state per batch and solve the accumulated normal equations once,
converging to the dense coefficients and intercepts within floating-point tolerance,
regardless of batch order or size. An in-memory execution intent for `fit_streaming` SHALL
instead collect the batches and reuse the dense path.

#### Scenario: Split batches equal the whole

- **WHEN** a dataset is fitted at once and, separately, its batches are accumulated with
  `partial_fit`
- **THEN** the two models' coefficients and intercepts agree within tolerance

#### Scenario: Uneven streaming batches match the dense fit

- **WHEN** a source emits uneven batches of data fitted densely
- **THEN** the streaming model's coefficients and intercepts agree with the dense fit within
  tolerance

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
- **THEN** it returns a structured error before touching the backend

#### Scenario: Negative penalty

- **WHEN** an estimator is built with a negative `alpha` and fitted
- **THEN** the fit returns an invalid-hyperparameter error naming `alpha`

#### Scenario: Prediction shape mismatch

- **WHEN** the model predicts on a matrix whose column count differs from the fitted design
- **THEN** it returns a shape-mismatch error

### Requirement: Backend and execution intent are carried and recorded

Per the `backend-selection` capability, every estimator and model operation SHALL record the
configured backend kind in its observability span; the solver actually used, being an
algorithm-level concept the shared span vocabulary does not carry, SHALL be exposed by the
fitted model. `fit` SHALL transfer the backend kind into the fitted model; the kind SHALL
never be hardcoded.

#### Scenario: Configured kind reaches the model

- **WHEN** an estimator is built with a non-default backend and fitted
- **THEN** the model reports that backend kind and its span records the kind, while
  `model.solver()` reports the resolved solver

### Requirement: The fitted model is shareable across threads

The fitted model SHALL be `Send + Sync` by construction so a single model serves concurrent
predictions without external synchronization.

#### Scenario: One model, many threads

- **WHEN** the same fitted model predicts on distinct inputs from multiple threads
- **THEN** every prediction matches the single-threaded result
