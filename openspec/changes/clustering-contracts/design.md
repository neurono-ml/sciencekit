# Design

## Context

See `proposal.md` — Why. `sciencekit_common` owns the fit and prediction contracts; the
current `SKUnsupervisedScorer` is bound to `SKRegressorPredictor` and rounds float
predictions to `usize`, which cannot represent noise. No concrete unsupervised scorer or
clusterer exists yet, so the bound can change before the first implementation ships.

## Goals / Non-Goals

**Goals:**
- One honest assignment contract shared by KMeans, MiniBatchKMeans, DBSCAN and future
  clusterers.
- An unsupervised scorer that consumes assignments directly.
- A documented noise convention.

**Non-Goals:**
- Any concrete clusterer or metric.
- Silhouette or other clustering metrics (the later metrics work).
- Changing the supervised scorer contracts.

## Decisions

### Decision 1 — New `SKClustererPredictor<F>` in the fit contracts

The trait lives beside `SKRegressorPredictor`/`SKClassifierPredictor`:

```rust
pub trait SKClustererPredictor<F: SKFloat> {
    type Error: From<SKError>;
    fn predict_assignments<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}
```

It is model-side only (no configured-estimator method), mirroring the other predictors.
The `i64` return is scalar-independent and leaves room for `-1` noise.

Alternative considered: reusing `SKClassifierPredictor` (wrong semantics: classifiers have
probabilities and canonical class tables) and keeping the regressor contract (cannot
represent noise and forces float encoding).

### Decision 2 — `SKUnsupervisedScorer` becomes generic over the clusterer contract

The pure form changes to `assignments: ArrayView1<'_, i64>`; the convenient form calls
`predict_assignments` and delegates. The trait bound becomes `M: SKClustererPredictor<F>`
and the existing `Self::Error: From<M::Error>` bound stays. This is a breaking change, but
no concrete implementation exists and the library is pre-1.0.

### Decision 3 — Noise policy belongs to the scorer

The contract does not silently drop `-1`; each scorer documents whether it excludes noise
rows or rejects them. A shared helper `sk_validate_assignments(features_rows, assignments)`
checks length and returns `ShapeMismatch`; the silhouette scorer will exclude noise, while
a future cluster-accuracy scorer can reject it.

### Decision 4 — Re-export and tests

`SKClustererPredictor` is re-exported from `sciencekit_common` and covered by the existing
scorer-contract tests with a constant clusterer. The tests prove the convenient form
delegates to the pure form, the noise label is visible and a length mismatch errors.

## Risks / Trade-offs

- **Breaking bound change** → documented as breaking; pre-1.0 and no implementors.
- **Noise semantics vary per metric** → the scorer documents its policy; the contract only
  reserves `-1`.
- **`i64` assignments are wider than `usize`** → intentional; cluster counts are small and
  noise needs a sentinel.

## Migration Plan

Update the trait, the re-export and the existing example tests in one change. Reverting
restores the regressor-bound scorer.

## Open Questions

None blocking. Silhouette and cluster-accuracy implementations belong to the later metrics
work.
