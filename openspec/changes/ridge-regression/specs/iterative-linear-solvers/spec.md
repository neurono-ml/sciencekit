## Purpose

Defines the iterative linear-solver substrate in `sciencekit_math`: a zero-copy operator
abstraction and an SPD conjugate-gradient solve, so a normal-equation system `AᵀA + αI` can
be solved without ever materializing `AᵀA`, and a sparse design can be fitted in place.

## ADDED Requirements

### Requirement: A zero-copy linear-operator abstraction

The library SHALL define a linear-operator trait generic over the sealed scalar, exposing the
operator's dimension and an application that writes the product of the operator and a vector
into a caller-provided output buffer, so an iterative solver drives an operator without owning
or copying its data.

#### Scenario: Applying an operator writes its product

- **WHEN** an operator is applied to a vector
- **THEN** the output buffer holds the product and the operator's borrowed data is unchanged

#### Scenario: Dimension is reported

- **WHEN** an operator is built over a square system of a given size
- **THEN** its reported dimension equals that size

### Requirement: The conjugate-gradient solve reaches tolerance

The library SHALL provide a conjugate-gradient solve for a symmetric positive-definite
operator and a right-hand side, returning the solution vector once the residual norm falls
below the requested tolerance or the iteration budget is exhausted. The iteration is a
sequential recurrence; the operator's application is where any parallelism lives.

#### Scenario: Small SPD system is solved

- **WHEN** a hand-checkable symmetric positive-definite system is solved
- **THEN** the residual norm of the returned solution is below the tolerance

#### Scenario: Matches a direct solve

- **WHEN** a well-conditioned SPD system is solved by conjugate gradient and by a direct
  solve
- **THEN** the two solutions agree within the tolerance

### Requirement: Non-convergence is structured, never silent

When the iteration budget is exhausted before the tolerance is met, the solve SHALL return a
structured non-convergence error carrying the number of iterations executed, and SHALL NOT
return a wrong answer as if it had converged.

#### Scenario: Zero iteration budget

- **WHEN** the solve is requested with an iteration budget of zero
- **THEN** it returns a non-convergence error reporting zero iterations

#### Scenario: Too tight a tolerance

- **WHEN** an unreachable tolerance is requested within a small iteration budget
- **THEN** the solve returns the non-convergence error rather than a stale vector

### Requirement: The sparse normal-equation operator never densifies

The library SHALL provide a zero-copy operator over a compressed sparse-row matrix that
applies `AᵀA` (or its `AᵀD`/`DA` factors) to a vector by walking the sparse structure in both
directions, without converting the sparse matrix to a dense one and without forming `AᵀA`.

#### Scenario: Product equals the explicit Gram product

- **WHEN** the sparse normal-equation operator applies `AᵀA` to a vector
- **THEN** the result equals the product of the explicitly formed `AᵀA` with that vector
  within tolerance

#### Scenario: The sparse matrix is borrowed, not copied

- **WHEN** the operator is constructed from a borrowed sparse view
- **THEN** the view's data is used in place and the operator holds no dense copy of it
