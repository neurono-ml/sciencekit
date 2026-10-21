## Purpose

Defines the shared coordinate-descent substrate of `sciencekit_linear_model`: the
objective configuration and its validation, the penalty-sequence conversion, the
cyclic/random sweep selection, the soft-threshold coordinate iteration over a dense or a
sparse design, the unpenalized centered intercept, the gap-safe screening with a KKT
reactivation guarantee, and the retained-design accumulator that backs the incremental and
out-of-core regimes of the L1-regularized estimators.

## ADDED Requirements

### Requirement: Objective configuration and validation

The substrate SHALL define a coordinate-descent configuration carrying `tolerance`,
`maximum_number_of_iterations`, the `positive` flag, the sweep `selection` and the
`screening` flag, plus the shared penalty-sequence conversion trait accepting a scalar `F`
and a sequence (`Vec<F>`, `Array1<F>`, `&[F]`, `[F; N]`). Configuration validation SHALL
reject a non-positive `tolerance`, a zero `maximum_number_of_iterations`, an empty penalty
sequence and any negative penalty with the central `SKError::InvalidHyperparameter`.

#### Scenario: Valid configuration is accepted

- **WHEN** a configuration is built with a positive tolerance, a positive iteration budget
  and a non-empty non-negative penalty sequence
- **THEN** it is accepted and exposes the configured values

#### Scenario: Invalid configuration is rejected

- **WHEN** the tolerance is zero, the iteration budget is zero, the penalty sequence is
  empty or any penalty is negative
- **THEN** validation returns an `InvalidHyperparameter` error naming the offending field

### Requirement: Soft-threshold coordinate iteration

The substrate SHALL solve the coordinate-descent objective
`(1/(2·n))‖y − Xβ‖₂² + α·ρ‖β‖₁ + (α·(1−ρ)/2)‖β‖₂²` by sweeping the coordinates and
updating each with a soft-threshold of the partial-residual gradient, and SHALL declare
convergence when the maximum coordinate change over a sweep falls to or below `tolerance`,
bounded by `maximum_number_of_iterations`; exhausting the budget without converging SHALL
raise `SKError::NotConverged { iterations }`.

#### Scenario: One-feature closed form

- **WHEN** a single centered feature is fitted with a scalar penalty
- **THEN** the coordinate equals the analytic soft-threshold solution within tolerance

#### Scenario: Orthogonal design closed form

- **WHEN** an orthogonal (uncorrelated) design is fitted
- **THEN** each coordinate equals its independent soft-threshold solution within tolerance

#### Scenario: Optimality (KKT) at the solution

- **WHEN** a fit has converged
- **THEN** every inactive coordinate has gradient magnitude at or below the L1 threshold and
  every active coordinate's signed gradient matches its L1/L2 subgradient within tolerance

#### Scenario: Exhausting the iteration budget

- **WHEN** the iteration budget is exhausted before convergence
- **THEN** the solve returns a non-convergence error carrying the iteration count

### Requirement: Dense and sparse designs share one iteration

The iteration SHALL run over a dense design and over a borrowed sparse CSR design through a
shared coordinate loop, and SHALL never densify a sparse design — the closest dense object
retained is the `O(n)` residual vector. A sparse design's stored entries are scanned to form
each column's signed dot and to update the residual.

#### Scenario: Sparse equals dense

- **WHEN** the same design is fitted densely and as an equivalent CSR matrix
- **THEN** the coefficient vectors agree within the documented relative tolerance

#### Scenario: The sparse design is not densified

- **WHEN** a sparse design is fitted
- **THEN** no `p × p` or `n × p` dense copy of the design is constructed

### Requirement: The intercept is never penalized

The substrate SHALL treat the intercept as an unpenalized coordinate updated to the residual
mean each sweep and SHALL run the feature updates on the mean-zero residual, algebraically
equivalent to centering the design; with the intercept disabled the solve SHALL run through
the origin.

#### Scenario: Penalty does not shrink the intercept

- **WHEN** data whose best line has a large intercept is fitted with growing penalty
- **THEN** the intercept stays near its unpenalized estimate while the slopes shrink toward
  zero

#### Scenario: Intercept disabled

- **WHEN** the same data is fitted without an intercept
- **THEN** the model reports a zero intercept and solves through the origin

### Requirement: Positive constraint and sweep selection

The substrate SHALL support an optional `positive` constraint clamping every coordinate at
zero after its soft-threshold update, and SHALL select the sweep order as either the cyclic
order or a deterministic pseudo-random permutation; the chosen order SHALL not change the
converged solution beyond tolerance.

#### Scenario: Positive constraint

- **WHEN** a design whose unconstrained solution has a negative coefficient is fitted with
  the positive constraint
- **THEN** the coefficient is clamped at zero and the remaining coordinates still satisfy the
  KKT conditions

#### Scenario: Random selection converges to the same solution

- **WHEN** the same problem is fitted under cyclic and under random selection
- **THEN** the two coefficient vectors agree within tolerance

### Requirement: Gap-safe screening is correct by construction

The substrate SHALL screen coordinates that provably cannot enter the active set: a strong
initial active set from the gradient at the origin, a gap-safe safe-screening elimination for
the pure-L1 objective derived from a dual-feasible point and the primal–dual gap radius, and a
KKT reactivation pass that adds back any coordinate whose full-gradient KKT condition is
violated. Screening SHALL never change the converged solution, and the fitted estimator SHALL
expose how many coordinates were screened.

#### Scenario: Screened equals unscreened

- **WHEN** the same problem is fitted with screening enabled and disabled
- **THEN** the two coefficient vectors agree within tolerance

#### Scenario: A wrongly screened coordinate is recovered

- **WHEN** a coordinate that belongs in the active set is deliberately screened out
- **THEN** the KKT reactivation pass adds it back and the solve returns the unscreened
  solution

### Requirement: The retained-design accumulator backs incremental and streaming fits

Because the L1 objective is not a function of the Gram matrix, the substrate SHALL retain the
design — dense rows appended, or sparse CSR batches stacked — together with the targets, and
SHALL re-solve at read time; the accumulated fit SHALL be independent of batch order and size.

#### Scenario: Accumulated equals concatenated

- **WHEN** batches of a design are accumulated and re-solved
- **THEN** the result equals the fit on the concatenated data within tolerance

#### Scenario: Empty accumulator

- **WHEN** no batch has been accumulated
- **THEN** a solve returns a structured error rather than a model

### Requirement: Independent per-alpha solves run in parallel

With several penalties configured, the substrate SHALL run the independent per-`alpha`
coordinate-descent solves over the compute pool, and the result SHALL be independent of the
thread count.

#### Scenario: Parallel equals sequential

- **WHEN** a multi-penalty problem is solved with the compute pool and with a single thread
- **THEN** the per-alpha coefficient rows agree within tolerance
