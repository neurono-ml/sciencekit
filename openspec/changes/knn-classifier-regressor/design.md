# Design

## Context

See `proposal.md` — Why. `sciencekit_neighbors` (W3.2) provides `SKKDTree`,
`SKBallTree` and the `SKNearestNeighborsSearcher` contract. `sparse-pairwise-distances`
provides the dense/sparse/mixed distance dispatcher. `sciencekit_common` provides the
builder state, label table, data/target views, errors and observability; `sciencekit_metrics`
provides the acceptance metrics. The estimators do not exist yet.

## Goals / Non-Goals

**Goals:**
- One model per estimator that works for dense, sparse and mixed inputs.
- A deterministic `Auto` policy that actually chooses a useful search strategy.
- Classification and regression semantics matching scikit-learn where the PRD is silent.
- Small, testable modules under the 200-line rule.

**Non-Goals:**
- Incremental (`partial_fit`) and out-of-core (`fit_streaming`) index maintenance.
- Radius-neighbour estimators and metric learning.
- `memmap2` backing, persistence and export (Wave 7).
- Approximate indexes.

## Decisions

### Decision 1 — Estimators live beside the indexes

`SKKNeighborsClassifier` and `SKKNeighborsRegressor` live in `sciencekit_neighbors`
because they are the first consumers of the tree/searcher modules and share their metric
enum. The fitted models own the training design (dense `Array2` or CSR `CsMat`) and,
optionally, a dense tree index. Owning the design is required because prediction must
outlive the caller's buffers and because sparse training has no tree.

Alternative considered: a separate estimators crate (rejected — circular dependency on
the index crate or an artificial split).

### Decision 2 — Runtime algorithm selection with a documented `Auto` policy

`SKNeighborsAlgorithm` is a closed enum (`Auto`, `KdTree`, `BallTree`, `BruteForce`).
`Auto` resolves at fit time:

```
dense && metric ∈ {Euclidean, Manhattan, Chebyshev} && n_samples >= 64
    → BallTree for n_features > 20, otherwise KdTree
sparse || mixed || metric == Cosine
    → BruteForce
```

The threshold is a documented engineering choice, not a statistical claim; an explicit
algorithm overrides it. The resolved strategy is recorded on the model and exposed as an
accessor for the tests. A metric/algorithm pair that cannot serve the representation
returns `UnsupportedRepresentation` at fit, before building anything.

Alternative considered: always building a tree (wasteful for small data and impossible
for sparse) and always brute force (ignores W3.2).

### Decision 3 — Brute-force searcher over the mixed distance dispatcher

The brute-force implementation calls `sk_pairwise_distance_matrix` on chunks of query
rows against the stored training design, then selects the `k` smallest `(distance, index)`
pairs with `select_nth_unstable_by` and sorts them. Chunk width is chosen from a fixed
memory budget, so a large prediction batch never materializes the full
`n_queries × n_training` matrix. Rows are independent and run through rayon; results are
collected in order.

Alternative considered: a per-pair manual loop (loses the SIMD/BLAS kernels and the
mixed-representation dispatch).

### Decision 4 — Voting, probabilities and distance weights

Classification counts weighted neighbour votes and selects the largest count; the class
tie-break is the smallest canonical index from `SKLabelTable`. Probabilities are the
weighted neighbour fractions, so they sum to one and include zero-probability classes.

`Distance` weights follow scikit-learn's convention: if any selected neighbour is at
distance zero, only the zero-distance neighbours receive weight and the others are
ignored; otherwise the weight is `1 / distance`. This avoids infinite weights and makes
duplicate training rows behave predictably. Regression uses the same weights and takes the
weighted mean of the responses.

Alternative considered: `1 / (distance + ε)` (simpler but changes the result for exact
duplicates and diverges from the reference).

### Decision 5 — Representation parity at the estimator level

Fit accepts dense, sparse or mixed input. Dense input may build a tree; sparse input and
mixed queries use the brute-force searcher over the stored design. Prediction accepts
dense, sparse or mixed queries; a dense tree is used only when the query is dense and the
metric is tree-compatible, otherwise the model falls back to brute force. No path densifies
a sparse input. This is the estimator-level parity required by
`representation-parity`; the index itself stays dense-only (W3.2).

### Decision 6 — Determinism and concurrency

Distance ties are ordered by `(distance, training_index)`; class ties by canonical class
index. Query rows are independent and the parallel path collects in row order. The fitted
model is immutable and `Send + Sync`, so concurrent prediction needs no locks.

### Decision 7 — Validation and observability

Builders expose `backend` (default `Faer`) and `execution_mode` (default `Automatic`);
fit transfers both into the model. `fit` and `predict_*` run inside the shared operation
span and record the backend. Errors use the central taxonomy: `EmptyInput`,
`InvalidHyperparameter` (`nearest_neighbors_count` zero or larger than the training set,
`leaf_size` zero), `ShapeMismatch`, `NonFiniteValue` (including a non-finite value in a sparse stored entry,
reported with its flat position), and `UnsupportedRepresentation`; `UnsupportedOperation`
appears only when the internal index builder is asked for the brute-force strategy, which
stores no index.

### Decision 8 — File layout

```
crates/sciencekit_neighbors/src/
  classifier/{mod.rs, builder.rs, core_implementation.rs, model.rs, fit.rs, predict.rs,
              targets.rs, voting.rs, *_tests.rs}
  regressor/{mod.rs, builder.rs, core_implementation.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  neighbors/{mod.rs, algorithm.rs, metric.rs, weights.rs, configuration.rs, index.rs,
             search.rs, training_data.rs, validation.rs, brute_force/, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; companion `*_tests.rs` modules are declared at the
end of their dispatcher.

## Risks / Trade-offs

- **Storing the training design costs `O(n · d)`** → inherent to a non-parametric
  estimator; sparse models store CSR, not a dense copy.
- **`Auto` threshold is heuristic** → exposed and tested; an explicit algorithm always
  wins.
- **Brute force is `O(n_queries · n_training)`** → chunked and parallel; the tree path
  covers the common dense case.
- **Distance weighting with duplicates** → the zero-distance convention keeps the result
  finite and matches the reference.
- **Cosine and NaN are not tree-compatible** → `Auto` routes them to brute force;
  explicit tree requests are rejected with guidance.
- **No `partial_fit`** → documented non-goal; a later out-of-core change can rebuild the
  index from a retained design.

## Migration Plan

Additive: new estimator types in an existing crate. No existing API changes.

## Open Questions

None blocking. Incremental index maintenance and approximate search are future changes.
