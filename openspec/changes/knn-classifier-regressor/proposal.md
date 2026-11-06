# Proposal

## Why

Wave 3's neighbours block needs user-facing estimators, not only spatial indexes. The
classifier and regressor are the canonical non-parametric baselines and the first
consumers of `SKKDTree`/`SKBallTree` (W3.2) and of the sparse/mixed distance dispatcher
(`sparse-pairwise-distances`). They also give `SKDBSCAN`, the imputer and later model
selection a reusable runtime search abstraction.

## What Changes

- New estimator types in `sciencekit_neighbors`: `SKKNeighborsClassifier<F>` and
  `SKKNeighborsRegressor<F>`, each with a distinct fitted model.
- Builder-only surface: `nearest_neighbors_count` (default 5), `weights` (`Uniform`
  default, `Distance`), `algorithm` (`Auto` default, `KdTree`, `BallTree`, `BruteForce`),
  `metric` (`Euclidean` default, `Manhattan`, `Chebyshev`, `Cosine`), `leaf_size`
  (default 40), `backend` (default `Faer`) and `execution_mode` (default `Automatic`).
- Runtime search dispatch: `Auto` uses a tree for dense Euclidean/Manhattan/Chebyshev
  data and brute force for sparse, mixed or cosine data; an explicit algorithm that
  cannot serve the representation returns `UnsupportedRepresentation`.
- Classification: majority vote, class-probability estimates as neighbour fractions,
  deterministic tie-breaking by the smallest canonical class index, integer and nominal
  targets through `SKLabelTable`.
- Regression: mean or inverse-distance-weighted mean of the neighbour responses.
- Representation parity for the estimator: dense, sparse and mixed fit/predict inputs;
  dense work may use a tree, sparse/mixed work uses the brute-force searcher over the
  stored training features; sparse input is never densified.
- Validation: empty input, zero or excessive neighbour count, feature-count mismatch,
  non-finite values, sparse NaN requests and unsupported metric/algorithm pairs.
- Explicit non-goals: `partial_fit`/`fit_streaming` (incremental index maintenance is a
  later out-of-core change), `memmap2` backing and model export (Wave 7).
- Companion tests: tiny and large data, every algorithm/metric/representation combination,
  deterministic ties, concurrent prediction and metric production.

No existing estimator API changes and no breaking changes.

## Capabilities

### New Capabilities

- `nearest-neighbor-estimators`: `SKKNeighborsClassifier` and
  `SKKNeighborsRegressor` — builder surface, runtime algorithm selection, voting and
  averaging, probability estimates, representation parity, determinism and validation.

### Modified Capabilities

None.

## Impact

- **Code:** `crates/sciencekit_neighbors` gains the estimator modules and models; no new
  crate.
- **APIs:** additive.
- **Dependencies:** `sciencekit_common`, `sciencekit_math`, `sciencekit_neighbors`'
  existing index and searcher modules; `sciencekit_metrics` as a dev-dependency for
  acceptance tests.
- **Downstream:** unblocks `SKKNNImputer` (W3.4), model selection (Wave 6) and gives
  `SKDBSCAN` a tested runtime searcher.
- **Acceptance criteria (PRD §8.7 / §10.3):** small and large data, concurrent prediction,
  metrics through `sciencekit_metrics`, companion `*_tests.rs` modules. Model export is
  deferred to Wave 7 and recorded in the PRD.
