# Conformance audit — implemented estimators against the parity primitives

This audit answers whether the algorithms already on `main` respect the parity doctrines
before the program scales to 150 algorithm changes. It covers the 48 implemented
estimator classes plus the in-flight, not-yet-implemented changes.

## 1. Verdict per doctrine

| Doctrine | State on main | Gap |
|---|---|---|
| `backend(SKBackendKind)` on every builder | **Respected** by all estimators | none |
| `execution_mode(SKExecutionMode)` on every builder | **Respected** by all estimators | none |
| Representation parity (dense/sparse/mixed, no densification) | **Respected** where the spec requires it; documented exceptions carry alternatives | none |
| Access-pattern declaration | **Absent**: ~25 sites hardcode `SKAccessPattern` inside operations; no estimator declares one | `access-pattern-contract` (already planned) |
| Streaming / lazy input | **Partial**: 17 estimators stream; KMeans, DBSCAN, KNN, trees, ensembles, SVM, PCA, TruncatedSVD, KNN imputer, missing indicator do not | `reopenable-fit-path` (new) + `mappable-fit-path` (planned) |
| Mappable input | **Absent**: no estimator consumes `SKMappableSource`; it exists only in `common`/`interop` | `mappable-fit-path` |
| `sample_weight` on fit | **Absent** from every public signature; trees compute weights internally and SGD synthesizes a vector of ones | `sample-weight-support` (new) |
| Code reuse of shared kernels | **Partial**: backend/solvers/pairwise/kernels are shared, but bootstrap sampling, histogram binning, incremental scatter and LBFGS live privately inside algorithm crates | promotion inside `resampling-primitives`, `binning-primitives`, `streaming-moments`, `lbfgs-optimizer` |
| Multiclass strategy | **Heterogeneous by design**: SVC uses OvO internally, LinearSVC OvR, logistic native multinomial/OvR, boosting native | refactor debt, not a parity gap; adoption happens inside `multiclass-strategy` |
| Model export | **Respected**: 47 fitted types export, 35 round-trip, refusals are typed | none |

## 2. Per-estimator scan

| Estimator | stream | partial_fit | sample_weight | export | backend | exec mode | inline pattern sites |
|---|---|---|---|---|---|---|---|
| KMeans | — | — | — (the k-means++ weighted draw is internal) | ✓ | ✓ | ✓ | 0 |
| MiniBatchKMeans | ✓ | ✓ | — | ✓ | ✓ | ✓ | 1 |
| DBSCAN | — | — | — | ✓ | ✓ | ✓ | 0 |
| PCA | — | — | — | ✓ | ✓ | ✓ | 2 |
| IncrementalPCA | ✓ | ✓ | — | ✓ | ✓ | ✓ | 3 |
| TruncatedSVD | — | — | — | ✓ | ✓ | ✓ | 1 |
| RandomForest | — | — | — | ✓ | ✓ | ✓ | 0 |
| GradientBoosting | — | — | — | ✓ | ✓ | ✓ | 0 |
| HistGradientBoosting | — | — | — | ✓ | ✓ | ✓ | 0 |
| Bagging | — | — | — | ✓ | ✓ | ✓ | 0 |
| IsolationForest | — | — | — | ✓ | ✓ | ✓ | 0 |
| SimpleImputer | ✓ | ✓ | — | ✓ | ✓ | ✓ | 2 |
| KNNImputer | — | — | — | ✓ | ✓ | ✓ | 0 |
| MissingIndicator | — | — | — | ✓ | ✓ | ✓ | 1 |
| LinearRegression | ✓ | ✓ | — | ✓ | ✓ | ✓ | 3 |
| Ridge | ✓ | ✓ | — | ✓ | ✓ | ✓ | 3 |
| Lasso | ✓ | ✓ | — | ✓ | ✓ | ✓ | 0 |
| ElasticNet | ✓ | ✓ | — | ✓ | ✓ | ✓ | 0 |
| LogisticRegression | ✓ | ✓ | — | ✓ | ✓ | ✓ | 1 |
| SGDClassifier | ✓ | ✓ | — (synthesized ones) | ✓ | ✓ | ✓ | 1 |
| SGDRegressor | ✓ | ✓ | — (synthesized ones) | ✓ | ✓ | ✓ | 1 |
| GridSearchCV / Halving | — | — | — | ✓ | ✓ | ✓ | 0 |
| KFold / StratifiedKFold / GroupKFold | n/a | n/a | n/a | n/a | — | ✓ | 0 |
| KNeighbors C/R | — | — | — | ✓ | ✓ | ✓ | 0 |
| KDTree / BallTree | n/a | n/a | n/a | n/a (data structures) | ✓ | ✓ | 0 |
| StandardScaler | ✓ | ✓ | — | ✓ | ✓ | ✓ | 3 |
| MinMaxScaler | ✓ | ✓ | — | ✓ | ✓ | ✓ | 3 |
| RobustScaler | ✓ | ✓ | — | ✓ | ✓ | ✓ | 2 |
| OneHotEncoder | ✓ | — | — | ✓ | ✓ | ✓ | 0 |
| PolynomialFeatures | ✓ | — | — | ✓ | ✓ | ✓ | 1 |
| SVC | — | — | — | ✓ | ✓ | ✓ | 0 |
| SVR | — | — | — | ✓ | ✓ | ✓ | 0 |
| LinearSVC | — | — | — | ✓ | ✓ | ✓ | 0 |
| DecisionTree C/R | — | — | internal only | ✓ | ✓ | ✓ | 0 |
| Pipeline | — | — | — | typed refusal | ✓ | ✓ | 0 |
| FeatureUnion | — | — | — | ✓ | ✓ | ✓ | 0 |

## 3. In-flight changes (planned, not implemented)

| Change | State | Audit finding |
|---|---|---|
| `outlier-lof-one-class` | 0/22 | declares streaming/`partial_fit`/`memmap2` non-goals although LOF is a `RandomAccess` (mappable) estimator; needs the declaration and a mappable fit; otherwise solid (SMO reuse, exports, parity) |
| `metrics-calibration-multilabel` | 0/21 | duplicates binning that `binning-primitives` will provide; the calibration curve must reuse the shared bins and state the tier of its inputs |
| `polars-interop` | 0/20 | reopenable source already planned; must also expose the mappable trait where positional access exists |
| `datafusion-interop` | 0/18 | same requirement as Polars |
| `jvm-dataset-interop` | 0/21 (deferred) | same requirement; deferral unchanged |
| `sparse-streaming-batches` | 27/28 | aligned; covers sparse batches for existing streaming estimators |
| `sparse-streaming-coverage` | 10/11 | aligned; audits sparse coverage of existing streaming estimators |
| `model-selection-core` | 12/13 | aligned; splitters are not estimators |

## 4. Actions taken from this audit

- **Edited (unimplemented):** `outlier-lof-one-class` gains the access-pattern
  declaration, a mappable fit and the metric-extension dependency; its memmap non-goal is
  restated. `metrics-calibration-multilabel` gains the binning-primitives dependency and
  the reuse requirement.
- **Created (retrofits for implemented estimators):**
  - `mappable-fit-path` — RandomAccess estimators accept `SKMappableSource`;
  - `reopenable-fit-path` — Iterative estimators gain `fit_streaming` over reopenable
    sources;
  - `sample-weight-support` — public per-sample weights across the estimators whose
    mathematics admits them.
- **Already covered by planned structural changes:** access-pattern declaration
  (`access-pattern-contract`), primitive promotions (`resampling-primitives`,
  `binning-primitives`, `streaming-moments`, `lbfgs-optimizer`), multiclass strategy
  (`multiclass-strategy`).
- **Recorded as debt, not a parity gap:** the heterogeneous internal multiclass
  strategies of SVC/LinearSVC/Logistic are adopted by `multiclass-strategy` without a
  behavior change.
