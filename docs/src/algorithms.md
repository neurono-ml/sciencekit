# Algorithms

The catalog mirrors the complete scikit-learn taxonomy. Every item enters the roadmap
with a mandatory four-stage evolution — **naive → tests → performance → out-of-core** —
and receives a Python binding as soon as it is validated on CPU.

<div class="sk-donut-wrap reveal">
  <div class="sk-donut">
    <svg viewBox="0 0 200 200" role="img" aria-label="Algorithm distribution by category">
      <circle cx="100" cy="100" r="80" stroke="#eceef1"/>
      <circle cx="100" cy="100" r="80" stroke="#0284c7" stroke-dasharray="62.83 502.65" stroke-dashoffset="-0.00"/>
      <circle cx="100" cy="100" r="80" stroke="#0d9488" stroke-dasharray="37.70 502.65" stroke-dashoffset="-62.83"/>
      <circle cx="100" cy="100" r="80" stroke="#7c3aed" stroke-dasharray="87.96 502.65" stroke-dashoffset="-100.53"/>
      <circle cx="100" cy="100" r="80" stroke="#4f46e5" stroke-dasharray="50.27 502.65" stroke-dashoffset="-188.50"/>
      <circle cx="100" cy="100" r="80" stroke="#e11d48" stroke-dasharray="37.70 502.65" stroke-dashoffset="-238.76"/>
      <circle cx="100" cy="100" r="80" stroke="#d97706" stroke-dasharray="25.13 502.65" stroke-dashoffset="-276.46"/>
      <circle cx="100" cy="100" r="80" stroke="#ea580c" stroke-dasharray="37.70 502.65" stroke-dashoffset="-301.59"/>
      <circle cx="100" cy="100" r="80" stroke="#059669" stroke-dasharray="37.70 502.65" stroke-dashoffset="-339.29"/>
      <circle cx="100" cy="100" r="80" stroke="#0891b2" stroke-dasharray="25.13 502.65" stroke-dashoffset="-376.99"/>
      <circle cx="100" cy="100" r="80" stroke="#db2777" stroke-dasharray="12.57 502.65" stroke-dashoffset="-402.12"/>
      <circle cx="100" cy="100" r="80" stroke="#2563eb" stroke-dasharray="37.70 502.65" stroke-dashoffset="-414.69"/>
      <circle cx="100" cy="100" r="80" stroke="#16a34a" stroke-dasharray="50.27 502.65" stroke-dashoffset="-452.39"/>
    </svg>
    <div class="sk-donut__center">
      <span class="sk-donut__value">40+</span>
      <span class="sk-donut__label">algorithms</span>
    </div>
  </div>
  <ul class="sk-donut__legend">
    <li><span class="sk-donut__swatch" style="background:#0284c7"></span>Preprocessing<span class="sk-donut__count">5</span></li>
    <li><span class="sk-donut__swatch" style="background:#0d9488"></span>Imputation<span class="sk-donut__count">3</span></li>
    <li><span class="sk-donut__swatch" style="background:#7c3aed"></span>Linear models<span class="sk-donut__count">7</span></li>
    <li><span class="sk-donut__swatch" style="background:#4f46e5"></span>Nearest neighbors<span class="sk-donut__count">4</span></li>
    <li><span class="sk-donut__swatch" style="background:#e11d48"></span>SVM<span class="sk-donut__count">3</span></li>
    <li><span class="sk-donut__swatch" style="background:#d97706"></span>Trees<span class="sk-donut__count">2</span></li>
    <li><span class="sk-donut__swatch" style="background:#ea580c"></span>Ensembles<span class="sk-donut__count">3</span></li>
    <li><span class="sk-donut__swatch" style="background:#059669"></span>Clustering<span class="sk-donut__count">3</span></li>
    <li><span class="sk-donut__swatch" style="background:#0891b2"></span>Decomposition<span class="sk-donut__count">2</span></li>
    <li><span class="sk-donut__swatch" style="background:#db2777"></span>Outliers<span class="sk-donut__count">1+</span></li>
    <li><span class="sk-donut__swatch" style="background:#2563eb"></span>Model selection<span class="sk-donut__count">3</span></li>
    <li><span class="sk-donut__swatch" style="background:#16a34a"></span>Metrics<span class="sk-donut__count">4+</span></li>
  </ul>
</div>

## Preprocessing · `sciencekit_preprocessing`

| Estimator | Description | Status |
|---|---|---|
| `SKStandardScaler` | Standardizes by removing the mean and scaling to unit variance (dense · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKMinMaxScaler` | Scales each feature to a fixed range | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKRobustScaler` | Scaling robust to outliers through quartile statistics | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKOneHotEncoder` | One-hot encoding of categories | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKPolynomialFeatures` | Polynomial and interaction features | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Imputation · `sciencekit_impute`

| Estimator | Description | Status |
|---|---|---|
| `SKSimpleImputer` | Mean, median, most frequent and constant imputation (dense · sparse · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKMissingIndicator` | `0.0`/`1.0` mask of the features missing at fit time (dense · sparse) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKKNNImputer` | Multivariate imputation via nearest neighbors | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Linear models · `sciencekit_linear_model`

| Estimator | Description | Status |
|---|---|---|
| `SKLinearRegression` | Ordinary least squares (dense · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKRidge` | Regression with L2 regularization (dense · sparse · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKLasso` | Regression with L1 regularization (dense · sparse · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKElasticNet` | Combined L1 + L2 regularization (dense · sparse · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKLogisticRegression` | Regularized logistic classification (binary · multinomial · one-vs-rest · dense · sparse · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKSGDClassifier` | Stochastic-gradient linear classification (5 losses · one-vs-rest · dense · sparse · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKSGDRegressor` | Stochastic-gradient linear regression (4 losses · Huber / epsilon-insensitive · dense · sparse · incremental · streaming) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKSGDClassifier` | Incremental classification via SGD · streaming (`SKLazySource`) | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKSGDRegressor` | Incremental regression via SGD · streaming (`SKLazySource`) | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Nearest neighbors · `sciencekit_neighbors`

| Structure/Estimator | Description | Status |
|---|---|---|
| `SKKNeighborsClassifier` | k-NN classification (dense trees · sparse brute force) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKKNeighborsRegressor` | k-NN regression (dense trees · sparse brute force) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKKDTree` | kd-tree for fast spatial search | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKBallTree` | Ball tree for arbitrary metrics | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Support vector machines · `sciencekit_svm`

| Estimator | Description | Status |
|---|---|---|
| `SKSVC` | Kernel-based C-support-vector classification (SMO, OvO, Platt + coupling, dense and sparse) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKSVR` | $\varepsilon$-insensitive support-vector regression (SMO, dense and sparse) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKLinearSVC` | Scalable linear SVM (dense and sparse) | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Trees and ensembles · `sciencekit_tree` + `sciencekit_ensemble`

| Estimator | Description | Status |
|---|---|---|
| `SKDecisionTreeClassifier` | CART for classification (dense and sparse, deterministic) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKDecisionTreeRegressor` | CART for regression (dense and sparse, deterministic) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKRandomForestClassifier` | Random-forest classification (bootstrap, OOB accuracy, dense and sparse) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKRandomForestRegressor` | Random-forest regression (bootstrap, OOB $R^2$, dense and sparse) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKBaggingClassifier` / `SKBaggingRegressor` | Generic parallel bootstrap and feature-subspace aggregation over any supervised base estimator (OOB accuracy / $R^2$, dense and sparse) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKGradientBoostingClassifier` / `SKGradientBoostingRegressor` | Sequential gradient boosting over negative gradients with Newton leaves, shrinkage, subsampling and early stopping (binary, multiclass, squared/absolute/Huber, dense and sparse) | <span class="sk-pill sk-pill--emerald">Implemented</span> |

## Unsupervised · `sciencekit_cluster`, `sciencekit_decomposition`, `sciencekit_outlier`

| Estimator | Description | Status |
|---|---|---|
| `SKKMeans` | Lloyd + Elkan k-means with seeded k-means++ (memmap indexes remain Phase 7) | <span class="sk-pill sk-pill--ok">Implemented</span> |
| `SKMiniBatchKMeans` | Mini-batch k-means with per-center learning rates, deterministic reassignment, `partial_fit` and multi-epoch streaming (`SKSourceProvider`) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKDBSCAN` | Density-based clustering over KD-tree / ball-tree and sparse brute-force region queries with explicit `SKClusterLabel::Noise` and nearest-core prediction | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKPCA` | Principal component analysis (full SVD · covariance eigh · sparse randomized SVD) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKTruncatedSVD` | Uncentered randomized low-rank SVD for latent semantic analysis and sparse dimensionality reduction (dense · sparse CSR) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKIncrementalPCA` | Out-of-core incremental PCA (Chan-Golub-LeVeque scatter merge · CSR Gram accumulation · `partial_fit` & `fit_streaming`) | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKIsolationForest` | Non-parametric anomaly detection via random isolation trees, harmonic path-length scoring, `Auto`/quantile contamination offsets and non-densifying CSR parity | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| Anomaly detection (LOF, One-Class SVM) | Remaining outlier estimators from the scikit-learn family | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Model selection, pipelines and metrics

| Item | Crate | Description | Status |
|---|---|---|---|
| `sk_train_test_split` | `sciencekit_model_selection` | Deterministic data partitioning | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKKFold` | `sciencekit_model_selection` | k-fold cross-validation | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKGridSearchCVClassifier` / `SKGridSearchCVRegressor` | `sciencekit_model_selection` | Typed grid search over configured candidates with shared folds | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKHalvingGridSearchClassifier` / `SKHalvingGridSearchRegressor` | `sciencekit_model_selection` | Successive-halving search: geometric resource schedule, seeded and stratified round subsamples, deterministic elimination trace, aggressive early pruning and winner refit | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| Halving resource axes beyond sample counts | `sciencekit_model_selection` | Additional `SKHalvingResource` variants (trees, epochs) | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKPipeline` | `sciencekit_pipeline` | Type-safe linear pipeline (compiler-checked composition) | <span class="sk-pill sk-pill--ok">Implemented</span> |
| `SKPipelineGraph` | `sciencekit_pipeline` | Validated runtime DAG pipeline: topological fit and predict, deterministic level order, edge-order merges without densifying CSR, node-attributed errors and classifier/regressor wrappers | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKStreamingPipeline` | `sciencekit_pipeline` | Out-of-core linear pipeline over an `SKSourceProvider`: chained `SKStreamingTransformer` fitting through the shared prefetch executor, one-shot materialization within a byte budget, multi-epoch replay, CSR-preserving per-batch transform and a bounded batch finalizer | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| Metrics foundation | `sciencekit_metrics` | Accuracy, precision/recall/F1 with macro/micro/weighted/binary averaging, confusion matrix, MSE/RMSE/MAE, R², scorer adapters and streaming accumulators | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| Full metrics | `sciencekit_metrics` | The remaining sklearn metrics (ROC/AUC, calibration, multilabel, ranking) | <span class="sk-pill sk-pill--neutral">Planned</span> |
| Extended Safetensors & JSON debug | `sciencekit_interop` | Binary Safetensors persistence (metadata header, `SKShardPlan` sharding, in-place header/payload padding, optional Gzip/Brotli/Snappy compression) and deterministic JSON debug inspection over `SKModelState` | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| Memory-mapped sources (`memmap` feature) | `sciencekit_interop` | Zero-copy `SKMappableSource`/`SKLazySource` over raw binary, NumPy `.npy` and footer-indexed Arrow IPC files, with descriptor, length, endianness, alignment and truncation validation | <span class="sk-pill sk-pill--emerald">Implemented</span> |
| `SKTelemetry` / `SKTelemetryPipeline` | `sciencekit_telemetry` | Opt-in OTLP trace, log and metric export pipelines, resource attributes, sampling and bounded lifecycle (silent no-op default) | <span class="sk-pill sk-pill--emerald">Implemented</span> |

<div class="sk-box sk-box--info">
<strong>Continuous acceleration:</strong> every completed algorithm receives, in separate changes,
a Python binding (<code>sciencekit_python</code>) and GPU backends (OpenCL → CUDA → ROCm) behind
<code>SKComputeBackend</code>. The CPU is the default backend, always present.
</div>

<div class="sk-btn-row">
  <a class="sk-btn sk-btn--primary" href="./architecture.html">Next: architecture →</a>
  <a class="sk-btn sk-btn--star" href="https://github.com/neurono-ml/sciencekit/issues">I want to implement one of these ⭐</a>
</div>
