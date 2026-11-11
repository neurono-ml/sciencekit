# Proposal

## Why

PCA and TruncatedSVD need a truncated singular-value decomposition, and `sciencekit_math`
only exposes a full SVD. The wave-plan candidate dependencies are now stale:
`rsvd-faer 0.1` depends on `faer 0.19` while the workspace uses `faer 0.24`, and the
`faer-sparse 0.17` line is not the monolithic `faer 0.24` dependency. A hand-rolled
randomized SVD over the existing `faer` backend avoids a duplicate linear-algebra stack.

## What Changes

- Add a `randomized-svd` kernel to `sciencekit_math`: a randomized range finder with
  power iterations, orthonormalization and a final small SVD, returning `U`, singular
  values and `V`.
- Accept dense and sparse feature views, dispatch once per operation and never densify a
  sparse input.
- Expose `number_of_components`, `number_of_power_iterations`, `random_seed` and an
  oversampling parameter; derive a deterministic random projection from the seeded
  generator.
- Fix a deterministic sign convention for singular vectors so repeated runs are identical.
- Validate component count, power iterations, oversampling and empty input with structured
  errors.
- No new external dependency; reuse `faer` for QR/GEMM and `sprs` for sparse products.
- Explicit non-goals: full SVD (already in the backend), ARPACK/IRLBA, mixed
  representation (a one-matrix kernel has only dense or sparse input), GPU.

## Capabilities

### New Capabilities

- `randomized-svd`: the deterministic randomized range finder and truncated SVD contract
  for dense and sparse matrices.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_math/src/decomposition/` (new module) and its re-exports.
- **APIs:** additive. A new result type and free `sk_` function.
- **Dependencies:** none new.
- **Downstream:** unblocks `SKPCA` (randomized solver) and `SKTruncatedSVD`.
- **Acceptance criteria:** dense/sparse equivalence, reconstruction accuracy, deterministic
  signs, large and small inputs, concurrent calls, companion `*_tests.rs` modules.
