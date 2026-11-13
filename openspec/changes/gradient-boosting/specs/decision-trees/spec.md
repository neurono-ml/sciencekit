# Spec Delta

## ADDED Requirements

### Requirement: Structure-only tree build and leaf replacement

The tree crate SHALL expose an advanced surface that grows a tree structure from a
response vector without committing to the public estimator's leaf-value policy, and that
replaces every leaf value with a caller-supplied value while preserving the split
structure.

#### Scenario: Boosting replaces leaf values

- **WHEN** gradient boosting grows a structure on pseudo-residuals and computes Newton
  leaf values
- **THEN** the crate replaces every leaf value with the supplied value and the split
  structure is unchanged

#### Scenario: Leaf replacement validates length

- **WHEN** the supplied leaf values do not match the number of leaves
- **THEN** the operation returns `ShapeMismatch` and leaves the structure unchanged

#### Scenario: Public estimator behavior is unchanged

- **WHEN** `SKDecisionTreeRegressor` is fitted through its public builder
- **THEN** its leaf values follow its documented criterion and the advanced surface is not
  required
