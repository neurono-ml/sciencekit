# Design

## Context

The weight plumbing already exists in the kernels: the tree splitter accumulates weighted
class counts, the coordinate-descent accumulator computes weighted Gram statistics, the
SGD epoch loop carries a weight vector (currently synthesized), and `sk_weighted_average`
exists in the common crate. What is missing is the public surface and the batch contract.

## Goals / Non-Goals

**Goals**
- One optional parameter with one validation rule everywhere.
- Streaming support, because out-of-core fits must not lose the weights.
- No unweighted-behavior drift.

**Non-Goals**
- Per-group weights, case weights distinct from sample weights, or weighting estimators
  whose mathematics does not admit them.

## Shape

```rust
pub fn fit<'a, X, Y, W>(&self, features: X, targets: Y, sample_weight: W)
    -> Result<Self::Model, Self::Error>
where
    X: TryInto<SKDataView<'a, F>, Error = SKError>,
    Y: TryInto<SKTargetView<'a>, Error = SKError>,
    W: Into<Option<ArrayView1<'a, F>>>;
```

`SKDataBatch` gains `weights: Option<Array1<F>>`; the prefetch driver passes it through
like the feature block. The SGD epoch loop reads the batch weights instead of
synthesizing ones.

## Per-family semantics

| Family | Where the weights land |
|---|---|
| scalers / imputer | weighted mean/variance/quantile accumulators (the shared moments) |
| linear / logistic / SGD | weighted Gram / gradient steps |
| SVM | box capacities scaled per sample |
| trees / ensembles | weighted impurity and weighted bootstrap frequencies |
| KMeans / MiniBatchKMeans | weighted centroids and inertia |
| metrics | weighted scoring entry points |

## Risks

- **Signature churn:** the parameter is optional and trailing; existing callers compile.
- **Hidden divergence:** the "ones equal absent" scenario pins the unweighted path.
