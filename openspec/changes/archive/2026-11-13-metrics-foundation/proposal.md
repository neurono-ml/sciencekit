# Proposal

## Why

Every estimator shipped in Waves 0–2 satisfies the functional criteria except one: the
library has no concrete metric. `sciencekit_common` defines the scoring contracts
(`SKLabelScorer`, `SKSupervisedScorer`, `SKUnsupervisedScorer`), but no crate implements
them, so fourteen estimators cannot be evaluated inside the library and the PRD §8.7
acceptance rule "produces metrics" is unmet by all of them. At the same time, the pairwise
catalogue stops at Euclidean, Manhattan and Cosine, while the next algorithms require two
more distances: Chebyshev (`p = ∞` neighbours) and a NaN-aware Euclidean — the only
distance a missing-value imputer can use, because it compares only co-observed features.
Closing both gaps now is the cheapest way to make every future algorithm evaluable and to
unblock `SKKNNImputer` without first building spatial-tree infrastructure.

## What Changes

- New sub-crate `sciencekit_metrics` with concrete classification metrics (accuracy,
  precision, recall, F1 with macro/micro/weighted averaging, confusion matrix) and
  regression metrics (mean squared error, root mean squared error, mean absolute error,
  coefficient of determination).
- Concrete scorer types implementing the existing `SKLabelScorer` and
  `SKSupervisedScorer` contracts, so any compatible fitted model is scored through the
  already-defined convenient/pure seam.
- Streaming accumulators for classification and regression, mergeable across threads and
  batches, producing the same values as the whole-input functions.
- `sciencekit_math::pairwise` gains a Chebyshev distance matrix and a NaN-aware Euclidean
  distance matrix (dense, borrowed inputs, explicit behaviour for pairs with no co-observed
  feature).
- Deterministic, documented label-space and averaging policy for multiclass metrics
  (observed-label union by default, explicit label list optional).
- Workspace membership and dependency wiring for the new crate.

No estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `evaluation-metrics`: concrete classification and regression metrics, their label-space
  and averaging policy, the scorer implementations over the shared scoring contracts, and
  the streaming accumulators.

### Modified Capabilities

- `pairwise-distances`: adds the Chebyshev distance and the NaN-aware Euclidean distance
  (co-observed-feature semantics) to the existing kernel catalogue.
- `backend-selection`: clarifies that the backend rule binds estimator and fitted-model
  operations; standalone metric and scoring functions are pure and carry no backend.
- `error-model`: adds precise variants for empty observation sets, non-finite numeric
  observations and labels outside the resolved label space, reusable across algorithms.

## Impact

- **Code:** new `crates/sciencekit_metrics` (workspace member); new modules under
  `crates/sciencekit_math/src/pairwise`; three additive variants on
  `sciencekit_common::SKError`; root `Cargo.toml` membership.
- **APIs:** additive only. Existing scorers remain trait-only; the new crate provides the
  first concrete implementations, plus free `sk_`-prefixed metric functions.
- **Dependencies:** `ndarray` and `rayon` from the existing workspace pins; no new external
  crate.
- **Downstream:** unblocks `SKKNNImputer` (`nan_euclidean`) and the evaluation stage of
  `SKKNeighborsClassifier/Regressor`, `SKSVC`, clustering and the later
  `SKGridSearchCV`; lets every shipped estimator satisfy the "produces metrics" half of the
  PRD §8.7 acceptance checklist.
- **Acceptance criteria (PRD §8.7 / §10.3):** correct results on small and large inputs;
  safe concurrent use (accumulators merge across rayon tasks); memory bounded by
  `O(classes²)` for classification and `O(1)` per regression accumulator; companion
  `*_tests.rs` modules. Model export stays out of scope because no `sciencekit_interop`
  crate exists yet; it is tracked for the interop change.
