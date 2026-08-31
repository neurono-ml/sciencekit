# error-model Specification

## Purpose
Establishes the central error taxonomy shared across the whole library: common variants named precisely, uniform conversion into algorithm-specific errors, and absence of ad-hoc error types in central contracts.
## Requirements
### Requirement: Central error with agreed variants
The library SHALL expose a central error enum covering at minimum: shape mismatch (with expected and found dimensions), data representation unsupported by the algorithm, invalid hyperparameter (identifiable), execution mode incompatible with the declared access pattern, non-convergence (with executed iteration count), input/output error and conversion failure at the data boundary.

#### Scenario: Shape mismatch identifies dimensions
- **WHEN** an operation receives data whose dimensions are incompatible with it
- **THEN** the produced error is the shape variant, carrying both expected and received shapes

#### Scenario: Unsupported representation is distinguished from invalid shape
- **WHEN** an algorithm operating only on dense data receives a sparse representation
- **THEN** the produced error is specifically unsupported-representation — not a generic shape error

#### Scenario: Non-convergence reports effort spent
- **WHEN** an iterative process exhausts its iterations without converging
- **THEN** the produced error reports the number of iterations executed

### Requirement: Uniform conversion for algorithm errors
Each algorithm crate SHALL be able to define its own error type, and that type SHALL convert from the central error automatically, keeping common errors identical across algorithms.

#### Scenario: Central error propagates through an algorithm error
- **WHEN** a consumer works with an algorithm's specific error type and a common library error occurs inside that algorithm's flow
- **THEN** the common error is converted automatically into the algorithm's type via the standard language conversion

### Requirement: I/O errors integrate into the taxonomy
Platform input/output errors SHALL convert into the central error without manual involvement from the caller.

#### Scenario: I/O failure becomes a library error
- **WHEN** a read/write operation fails with a platform error during processing
- **THEN** the consumer receives the central error in the I/O variant, with the original error preserved as source

