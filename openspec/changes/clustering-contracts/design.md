# Design

## Context

See `proposal.md` — Why. `sciencekit_common` owns the fit and prediction contracts; the
current `SKUnsupervisedScorer` is bound to `SKRegressorPredictor` and rounds float
predictions to `usize`, which cannot represent noise. No concrete unsupervised scorer or
clusterer exists yet, so the bound can change before the first implementation ships.

## Goals / Non-Goals

**Goals:**
- One honest, typed assignment contract shared by KMeans, MiniBatchKMeans, DBSCAN and
  future clusterers.
- A noise representation that cannot collide with a cluster identifier.
- An unsupervised scorer that consumes typed assignments directly.
- A compact representation (at most four bytes per element).

**Non-Goals:**
- Any concrete clusterer or metric.
- Silhouette or other clustering metrics (the later metrics work).
- Changing the supervised scorer contracts.

## Decisions

### Decision 1 — A typed `SKClusterLabel` with a niche-based identifier

```rust
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum SKClusterLabel {
    Noise,
    Cluster(SKClusterIdentifier) — a newtype over `NonZeroU32` storing `index + 1`,
}
```

`NonZeroU32` gives the enum a zero niche, so `Noise` uses the otherwise-invalid zero value
and the enum stays four bytes. Cluster index `0` is stored as `NonZeroU32::new(1)`;
conversion helpers subtract one at the boundary. This is safer than `i64` with `-1` and
smaller than an enum with a plain `u32` (which needs a separate discriminant and becomes
eight bytes).

Alternative considered: `Option<NonZeroU32>` (equally compact but less expressive at call
sites) and `i64` with `-1` (scikit-learn compatibility, but a sentinel that can leak into
indexing). The typed enum wins on safety without a memory penalty.

### Decision 2 — `SKClustererPredictor<F>` returns typed labels

The trait lives beside `SKRegressorPredictor`/`SKClassifierPredictor`:

```rust
pub trait SKClustererPredictor<F: SKFloat> {
    type Error: From<SKError>;
    fn predict_clusters<'a, X>(
        &self,
        features: X,
    ) -> Result<Array1<SKClusterLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}
```

It is model-side only, mirroring the other predictors. KMeans/MiniBatch only produce
`Cluster`; DBSCAN produces both variants.

### Decision 3 — `SKUnsupervisedScorer` consumes typed assignments

The pure form changes to `assignments: ArrayView1<'_, SKClusterLabel>`; the convenient
form calls `predict_clusters` and delegates. The trait bound becomes
`M: SKClustererPredictor<F>` and the existing `Self::Error: From<M::Error>` bound stays.
This is a breaking change, but no concrete implementation exists and the library is
pre-1.0.

### Decision 4 — Noise policy belongs to the scorer

The contract does not silently drop noise; each scorer documents whether it excludes noise
rows or rejects them. A shared helper `sk_validate_cluster_assignments(number_of_rows, assignments)`
checks length and returns `ShapeMismatch`; the silhouette scorer will exclude noise, while
a future cluster-accuracy scorer can reject it.

### Decision 5 — Interoperability stays at the boundary

`SKClusterLabel` implements `to_i64`/`from_i64` mapping `Noise ↔ -1` and `Cluster(id) ↔
id`. Only Python/ONNX/CSV adapters use the integer convention; the Rust API never sees a
sentinel. `from_i64` rejects values below `-1` with `InvalidHyperparameter` (or a
conversion error).

### Decision 6 — Re-export and tests

`SKClusterLabel` and `SKClustererPredictor` are re-exported from `sciencekit_common`. Tests
cover: a `ConstantClusterer` (clusters only) and a `MixedClusterer` stand-in that emits noise, the four-byte size assertion, the
conversion round-trip and invalid values, the convenient scorer form, and a length
mismatch.

## Risks / Trade-offs

- **Breaking bound change** → documented as breaking; pre-1.0 and no implementors.
- **NonZero encoding requires `id + 1`** → conversion helpers centralize it and tests
  cover cluster `0`.
- **Four-byte assumption** → asserted by a `size_of` test; if a future Rust layout changes,
  the test fails loudly.
- **Interop needs conversion** → a single pair of helpers, used only at the boundary.
- **Noise semantics vary per metric** → the scorer documents its policy; the contract only
  models the variant.

## Migration Plan

Update the trait, the re-export and the existing example tests in one change. Reverting
restores the regressor-bound scorer.

## Open Questions

None blocking. Silhouette and cluster-accuracy implementations belong to the later metrics
work.
