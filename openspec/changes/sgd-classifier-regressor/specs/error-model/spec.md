# Spec Delta

## MODIFIED Requirements

### Requirement: Central error with agreed variants
The library SHALL expose a central error enum covering at minimum: shape mismatch (with expected and found dimensions), data representation unsupported by the algorithm, invalid hyperparameter (identifiable), execution mode incompatible with the declared access pattern, non-convergence (with executed iteration count), input/output error, conversion failure at the data boundary, and an operation unsupported by an otherwise valid configuration (naming the operation and the reason).

#### Scenario: Shape mismatch identifies dimensions
- **WHEN** an operation receives data whose dimensions are incompatible with it
- **THEN** the produced error is the shape variant, carrying both expected and received shapes

#### Scenario: Unsupported representation is distinguished from invalid shape
- **WHEN** an algorithm operating only on dense data receives a sparse representation
- **THEN** the produced error is specifically unsupported-representation — not a generic shape error

#### Scenario: Non-convergence reports effort spent
- **WHEN** an iterative process exhausts its iterations without converging
- **THEN** the produced error reports the number of iterations executed

#### Scenario: Unsupported operation names the operation and the reason
- **WHEN** a validly configured model is asked for an output its configuration cannot produce
- **THEN** the produced error names the requested operation and the reason it is unavailable
