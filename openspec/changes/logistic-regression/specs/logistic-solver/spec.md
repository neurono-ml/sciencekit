## Purpose

Defines the shared logistic solver substrate of `sciencekit_linear_model`: the objective
configuration and its validation, the penalty/solver/multi-class resolution, the class-weight
normalization, the pure-Rust L-BFGS-B, SAGA and LIBLINEAR dual coordinate-descent solvers,
the shared dense/sparse design abstraction and the retained-design accumulator that backs the
incremental and out-of-core regimes of the linear classifier.

## ADDED Requirements

### Requirement: Objective configuration and validation

The substrate SHALL define a logistic configuration carrying `tolerance`,
`maximum_number_of_iterations`, the `penalty`, `l1_ratio`, the `solver`, the `multi_class`
strategy, the `class_weight` and the `include_intercept` flag. Validation SHALL reject a
non-positive `tolerance`, a zero `maximum_number_of_iterations`, a non-positive
regularization strength and an `l1_ratio` outside `[0, 1]` with the central
`SKError::InvalidHyperparameter`.

#### Scenario: Valid configuration is accepted

- **WHEN** a configuration is built with a positive tolerance, a positive iteration budget, a
  positive regularization strength and an in-range `l1_ratio`
- **THEN** it is accepted and exposes the configured values

#### Scenario: Invalid configuration is rejected

- **WHEN** the tolerance is zero, the iteration budget is zero, the regularization strength
  is non-positive or `l1_ratio` is out of range
- **THEN** validation returns an `InvalidHyperparameter` error naming the offending field

### Requirement: Penalty and solver compatibility is resolved before the solve

The substrate SHALL enforce scikit-learn's solver/penalty matrix: `Lbfgs` supports `L2` and
`None`; `Liblinear` supports `L1` and `L2`; `Saga` supports `L1`, `L2`, `ElasticNet` and
`None`. An unsupported combination SHALL raise a structured error naming both the solver and
the penalty. `l1_ratio` SHALL be honored only with `ElasticNet` and SHALL otherwise be
ignored.

#### Scenario: Unsupported solver and penalty combination

- **WHEN** a configuration pairs the `Lbfgs` solver with the `L1` penalty
- **THEN** the fit returns a structured error naming the solver and the penalty

#### Scenario: Supported combinations solve

- **WHEN** each supported solver/penalty pair is fitted on a small design
- **THEN** every pair converges and produces a finite solution

### Requirement: Class weights are normalized per sample

The substrate SHALL accept an optional class weighting and SHALL expand it into one
non-negative per-sample weight `c_i` that multiplies that sample's loss term, leaving the
penalty unweighted. The `Balanced` policy SHALL weight each class by
`n_samples / (n_classes · class_count)`. A negative or all-zero explicit weight SHALL be
rejected with a structured error.

#### Scenario: Balanced weighting balances the classes

- **WHEN** a heavily imbalanced design is fitted with the `Balanced` policy
- **THEN** each class's total weight is equal within tolerance

#### Scenario: Negative weight is rejected

- **WHEN** an explicit class weight is negative
- **THEN** the fit returns a structured error naming the class weight

### Requirement: The L-BFGS-B solver solves the smooth path

The substrate SHALL solve the smooth `L2`/`None` objective with a limited-memory BFGS
quasi-Newton iteration that maintains a bounded history of curvature pairs, takes an Armijo
backtracking line search, converges on the gradient infinity norm against `tolerance`,
bounded by `maximum_number_of_iterations`, and raises `SKError::NotConverged { iterations }`
when the budget is exhausted. (Because the solver/penalty matrix forbids an L1 penalty for
this solver, no orthant-wise step is required.)

#### Scenario: Smooth L2 fit converges

- **WHEN** a well-separated binary design is fitted with the `Lbfgs` solver and an L2 penalty
- **THEN** the solve converges within the budget and the gradient norm is at or below tolerance

#### Scenario: Exhausting the iteration budget

- **WHEN** the iteration budget is exhausted before convergence
- **THEN** the solve returns a non-convergence error carrying the iteration count

### Requirement: SAGA solves the non-smooth and sparse paths

The substrate SHALL solve the `L1`, `ElasticNet` and `None` objectives with a
variance-reduced incremental-gradient (SAGA) iteration that maintains a per-sample gradient
table, applies the proximal soft-threshold step for the L1 component, and converges on the
duality-gap or gradient criterion against `tolerance`, bounded by
`maximum_number_of_iterations`. The SAGA path SHALL consume a borrowed CSR design without
densifying it.

#### Scenario: L1 fit produces exact zeros

- **WHEN** a design with irrelevant features is fitted with the `Saga` solver and an L1
  penalty
- **THEN** the irrelevant coefficients are exactly zero

#### Scenario: SAGA never densifies a sparse design

- **WHEN** a borrowed CSR design is fitted with the `Saga` solver
- **THEN** no dense `n × p` copy of the design is constructed

### Requirement: The LIBLINEAR coordinate-descent solver solves the one-vs-rest path

The substrate SHALL provide a LIBLINEAR-style solver for the `L1` and `L2` penalties that
operates one-vs-rest: for `L2` it updates the box-constrained dual coordinates, each by a
Newton step on the scalar subproblem; for `L1` it runs a primal coordinate descent with a
soft-threshold step. Both converge on the maximum coordinate change against `tolerance`,
bounded by `maximum_number_of_iterations`. As in scikit-learn's LIBLINEAR path, the intercept
is augmented as a constant feature and is therefore regularised.

#### Scenario: Coordinate-descent solver matches the primal optimum

- **WHEN** the same binary design without an intercept is fitted with the `Liblinear` solver
  and with the `Lbfgs` solver under an L2 penalty
- **THEN** the two coefficient vectors agree within tolerance

#### Scenario: Dual solver keeps sparse designs sparse

- **WHEN** a borrowed CSR design is fitted with the `Liblinear` solver
- **THEN** no dense `n × p` copy of the design is constructed

### Requirement: Dense and sparse designs share one solver interface

The substrate SHALL expose a shared design abstraction over a dense `ArrayView2` and a
borrowed `CsMatView`, over which the solvers form matrix-vector products and per-sample
gradients, and SHALL never densify a sparse design; the only dense object retained for a
sparse design is the `O(n · n_classes)` linear predictor.

#### Scenario: Sparse equals dense

- **WHEN** the same design is fitted densely and as an equivalent CSR matrix with the same
  solver and penalty
- **THEN** the coefficients and intercepts agree within the documented relative tolerance

### Requirement: The retained-design accumulator backs incremental and streaming fits

Because the logistic objective is not a function of the Gram matrix, the substrate SHALL
retain the design — dense rows appended, or sparse CSR batches stacked — together with the
canonical labels, and SHALL re-solve at read time (the per-sample weights are re-derived from
the class-weighting policy at each solve); the accumulated fit SHALL be independent of batch
order and size.

#### Scenario: Accumulated equals concatenated

- **WHEN** batches of a design are accumulated and re-solved
- **THEN** the result equals the fit on the concatenated data within tolerance

#### Scenario: Empty accumulator

- **WHEN** no batch has been accumulated
- **THEN** a solve returns a structured error rather than a model
