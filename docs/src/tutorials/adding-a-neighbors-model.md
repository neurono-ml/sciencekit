# Adding a Neighbors Model — `SKKNeighborsClassifier` and `SKKNeighborsRegressor`

> Pre-read: [How to add an Algorithm](how-to-add-an-algorithm.md) and the
> [k-nearest neighbours chapter](../development-explained/ch32-knn-estimators.html).

This tutorial walks through the neighbours estimators as they are **shipped**: what `fit`
stores, how `predict` searches, why the representation chooses the strategy, and the exact
public surface you compose in a pipeline or a grid search.

Two rules shape everything below:

- `fit` stores the training design and the label table; it does not compress or discard rows.
- `predict` needs **the nearest stored rows to each query**, so the estimator chooses a search
  strategy at run time: a spatial index for dense low-dimensional data, brute force for sparse,
  mixed or cosine data.

## 1. The public surface

```rust
use sciencekit_neighbors::{SKKNeighborsClassifier, SKNeighborsWeights};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

let classifier = SKKNeighborsClassifier::<f64>::build()
    .nearest_neighbors_count(5)
    .weights(SKNeighborsWeights::Uniform) // or Distance
    .build()?;

// `labels` is an `Array1<i64>`; nominal labels can be a `Vec<&str>` / `&[&str]`.
let labels = ndarray::Array1::from_vec(vec![0_i64, 1, 0, 1]);
let model = classifier.fit(training_features.view(), labels.view())?;
let labels_out = model.predict_labels(query_features.view())?;
let probabilities = model.predict_probabilities(query_features.view())?;
```

The builder is the only constructor. It exposes:

| Setter | Default | Meaning |
|---|---|---|
| `nearest_neighbors_count` | 5 | how many neighbours vote |
| `weights` | `Uniform` | `Distance` uses inverse-distance weighting |
| `algorithm` | `Auto` | `KdTree`, `BallTree`, `BruteForce` or automatic |
| `metric` | Euclidean | Euclidean, Manhattan, Chebyshev, Cosine |
| `leaf_size` | 40 | tree leaf bound (dense tree paths) |
| `backend` | `Faer` | carried into the model and its spans |
| `execution_mode` | `Automatic` | recorded configuration |

The regressor (`SKKNeighborsRegressor`) has the same surface and returns responses.

## 2. What `fit` stores

`fit` keeps:

- the **training design** as an owned dense block or a CSR matrix — never a copy per query;
- the **label table** for classification (`SKLabelTable`), built in first-occurrence order, the
  library-wide convention;
- the configuration, including the resolved search strategy.

A fitted classifier exposes `classes()` (the label table), the resolved strategy and the stored
representation; the regressor exposes the same minus the table.

## 3. How `predict` searches

`Auto` resolves once, when the model is fitted:

| Data | Metric | Strategy |
|---|---|---|
| dense, **$\ge 64$ training rows**, $\le 20$ features | Euclidean/Manhattan/Chebyshev | k-d tree |
| dense, $\ge 64$ rows, $> 20$ features | Euclidean/Manhattan/Chebyshev | ball tree |
| dense, < 64 rows | any | brute force (a tree costs more than it saves) |
| sparse or mixed | any | brute force over stored entries |
| any | Cosine | brute force (cosine has no tree-pruning form) |

An **explicit** `KdTree` or `BallTree` request on a representation the index cannot serve returns
`UnsupportedRepresentation` with a suggestion to use `Auto` or `BruteForce`. Sparse input is
never densified.

The brute-force searcher and the trees share one contract, so the neighbour list is always
ordered by **distance, then the smallest training index** — the determinism guarantee that the
vote and the probabilities rely on.

## 4. Voting and probabilities

Classification takes the majority of the $k$ neighbours. With `Distance` weighting each neighbour
contributes $1/d$ (a zero-distance neighbour uses the documented convention, so weights stay
finite). Probabilities are the weighted neighbour fractions and sum to one; the columns follow
the class table. A vote tie goes to the **smallest canonical class index**.

Regression returns the mean of the neighbour responses (uniform) or their inverse-distance
weighted mean. Integer targets are promoted to the model scalar.

## 5. What to test when you add a neighbour variant

Follow the companion-module rule: put the tests beside the implementation.

1. **Dense/sparse parity** — the same logical design gives the same predictions.
2. **Brute-force agreement** — the new path returns exactly what the trees return.
3. **Ties** — equidistant neighbours and tied votes resolve deterministically.
4. **Representation rejection** — an explicit unsupported tree request is a structured error.
5. **Concurrency** — a fitted model is `Send + Sync` and parallel prediction equals sequential.

## 6. Out-of-core today

The memory-mapped base (`SKMappableSource`, `memmap2`) and model persistence/export are
**Phase 7 (interop)** work, not part of the neighbours change. Streaming `fit_streaming` and
radius-based estimators are explicit non-goals of this change; the estimators hold the training
design in memory. The PRD records both deferrals in the Phase 3 section.
