# Design

## Context

The parity program is a planning artifact plus a graph extension. This design fixes the
matrix schema, the granularity rule for algorithm changes, the memory-tier doctrine that
the matrix references, the wave order, and how the matrix stays in sync with
`implementation-dag/dag.yaml`. It does not design any estimator.

## Goals / Non-Goals

**Goals**

- One row per baseline API item, one status, one tier, one owner.
- One change per algorithm family, so history maps to algorithms instead of batches.
- Structural work (access pattern, sample weights, multiclass and multioutput strategy,
  estimator tags) lands before the algorithms that depend on it.
- Waves that maximize safe parallelism without two changes editing the same crate.

**Non-Goals**

- Implementing any estimator in this change.
- Reproducing scikit-learn internals, private APIs, plotting helpers or `utils`.
- Promising out-of-core execution for mathematics that cannot provide it (see tiers).

## Matrix schema

`parity.yaml`:

```yaml
baseline: "1.9.1"
reference: data/sklearn-1.9.1-api.txt
crates:
  - sciencekit_preprocessing
  - ...
entries:
  - api: sklearn.cluster.AgglomerativeClustering
    family: cluster
    kind: estimator          # estimator | transformer | function-family
    status: planned          # implemented | planned | absent
    crate: sciencekit_cluster
    change: agglomerative-clustering
    tier: full-memory        # streaming | mappable | chunked | full-memory
    reuses: [pairwise-metric-extension, sparse-symmetric-eigensolver]
    alternative: null        # required when tier == full-memory
    reason: null             # required when status == absent
```

`kind: function-family` rows cover the ~300 baseline functions grouped by capability
(for example `sklearn.metrics` regression metrics, `sklearn.datasets` generators), so the
matrix stays readable without pretending each free function is a change.

## Granularity rule: one algorithm per change

The unit is the **algorithm or solver**, not the Python class:

- Variants that share one mathematical core and one solver live in the same change:
  `lars-family` covers `Lars`, `LassoLars`, `LarsCV`, `LassoLarsCV`, `LassoLarsIC`;
  `multitask-linear` covers the multi-task Lasso and Elastic Net and their CV wrappers;
  `univariate-selectors` covers `SelectKBest`, `SelectPercentile`, `SelectFpr`, `SelectFdr`,
  `SelectFwe` and `GenericUnivariateSelect`, which differ only by the selection rule.
- A variant that changes the solver or the memory profile gets its own change:
  `nmf` and `minibatch-nmf`, `dictionary-learning` and `minibatch-dictionary-learning`,
  `sparse-pca` and `minibatch-sparse-pca`, `mds` and `classical-mds` are separate because
  the online/randomized/classical formulations are different algorithms.
- Pure wrappers are separate changes only when they add scheduling or selection
  machinery: `ridge-cv`, `lasso-cv`, `elastic-net-cv`, `randomized-search`,
  `halving-random-search`.
- Trivial estimators that exist only to establish a baseline still get one change each
  (`dummy-estimators` covers both Dummy models because they share one trivial strategy).

Every change in the program declares exactly one matrix owner row per class it delivers,
except the grouped variants above, which declare several.

## Memory tiers

The matrix's `tier` field anticipates the `access-pattern-contract` requirement:

| Tier | Meaning | Typical families |
|---|---|---|
| `streaming` | single pass or `partial_fit`, constant model memory | naive Bayes, SGD variants, MLP/RBM, GMM, minibatch factorizations, BIRCH, vectorizers, encoders, scalers |
| `mappable` | O(n) storage on disk with random access | KNN family, KernelDensity, trees and forests, bagging, isolation forest, KMeans, DBSCAN, MeanShift, OPTICS/HDBSCAN, spectral over a kNN graph |
| `chunked` | multi-pass over batches with O(p²) or O(k·n) accumulators | CV linear models, GLMs, Huber, LARS, OMP, NMF, FastICA, FactorAnalysis, LDA/QDA, covariance, CCA/PLS, feature selection, calibration, inspection utilities |
| `full-memory` | O(n²) or O(n³); needs a documented alternative | Agglomerative/FeatureAgglomeration, Spectral{Biclustering,Coclustering}, AffinityPropagation, MDS/Isomap/LLE/t-SNE, KernelPCA, exact KernelRidge, Gaussian Processes, NCA |

`full-memory` is admissible only with a named alternative (Nystroem, RandomizedSVD,
coreset/mini-batch, sparse kNN affinity graph) and a typed refusal when the budget is
exceeded — the memory analogue of the `data-view-boundary` dense-only exception.

## Wave order

Structural changes first; then cheap high-coverage families; then the heavy model
families; the DAG lanes govern actual parallelism, the waves are the narrative order.

| Wave | Content | Blocking structural dependencies |
|---|---|---|
| 8 | parity matrix, access-pattern contract, mappable fit path, sample weights, multiclass and multioutput strategy, estimator tags, dataset generators/loaders, and the 16 shared primitives of the reuse map | none |
| 9 | preprocessing completion, feature selection, naive Bayes, dummy, isotonic, metrics completion | access pattern, sample weights |
| 10 | linear-model completion, model-selection completion | structure + wave 9 encoders |
| 11 | kernel methods, GP, neural networks, semi-supervised, text vectorizers, compose, ensemble extras, calibration, inspection | structure + wave 10 selection |
| 12 | cluster, decomposition, mixture, discriminant, covariance, cross-decomposition | structure + parallel distances + SVD |
| 13 | manifold, radius/kernel neighbors, random projection, threshold tuning | structure + wave 12 |

## Reuse before implementation

`reuse-map.md` is normative: it inventories the primitives the core already provides
(backend, solvers, pairwise, kernels, batching, coordinate descent, neighbor searchers,
tree splitter, accumulators, Platt scaling, interop sources) and the sixteen primitives
that must be promoted or created before the algorithm waves. Four already exist privately
and are promoted rather than rewritten: bootstrap sampling, histogram binning, incremental
scatter and LBFGS.

The rules: every algorithm proposal names the primitives it consumes; a primitive changes
only through its own foundations change; a helper is promoted when a second consumer
appears and generalized at the third; each primitive change refactors its existing private
home without changing results; the matrix row carries the `reuses` list, checked by the
matrix checker. This is what keeps 160+ algorithm changes from re-implementing the same
solver, sampler, metric or accumulator.

## DAG integration

- New lanes: `planning`, `contracts`, `datasets`, `preprocessing`, `feature-selection`,
  `naive-bayes`, `linear-model`, `kernel`, `neural-network`, `semi-supervised`, `text`,
  `compose`, `ensemble`, `meta`, `mixture`, `discriminant`, `covariance`,
  `cross-decomposition`, `manifold`, `inspection`, `calibration`.
- The algorithm changes append to the existing `neighbors`, `cluster`, `decomposition`,
  `model-selection` and `metrics` lanes.
- Every algorithm change has a hard dependency on `access-pattern-contract`; classifiers
  that are binary-only also depend on `multiclass-strategy`; estimators that accept
  weights depend on `sample-weight-support`.
- Join changes (`model-export-retrofit`, `onnx-export`, `umbrella-crate`,
  `python-bindings-complete`) gain the new model families as hard dependencies **only**
  when the family lands; the DAG records them progressively, one family at a time, so the
  joins never block the program.

## Risks

- **Moving target:** pinning 1.9.1 freezes the goal; upgrading the baseline is itself a
  change and adds rows as `planned`/`absent`.
- **Function tier drift:** the ~300 functions are grouped; a new function in an existing
  family is covered by that family's change, a new family needs a row.
- **Full-memory honesty:** some families cannot be out-of-core. The doctrine requires the
  alternative path and a typed refusal instead of an OOM.
- **DAG size:** ~167 new nodes (150 algorithm/gap changes plus 17 foundations). The consistency checker and the readiness script are the
  only way to keep it usable; the design tables are a cache, `dag.yaml` is the source.
