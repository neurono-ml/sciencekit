# Algorithms

The catalog mirrors the complete scikit-learn taxonomy. Every item enters the roadmap
with a mandatory four-stage evolution — **naive → tests → performance → out-of-core** —
and receives a Python binding as soon as it is validated on CPU.

<div class="sk-donut-wrap reveal">
  <div class="sk-donut">
    <svg viewBox="0 0 200 200" role="img" aria-label="Algorithm distribution by category">
      <circle cx="100" cy="100" r="80" stroke="#eceef1"/>
      <circle cx="100" cy="100" r="80" stroke="#0284c7" stroke-dasharray="64.44 502.65" stroke-dashoffset="0"/>
      <circle cx="100" cy="100" r="80" stroke="#0d9488" stroke-dasharray="25.78 502.65" stroke-dashoffset="-64.44"/>
      <circle cx="100" cy="100" r="80" stroke="#7c3aed" stroke-dasharray="90.22 502.65" stroke-dashoffset="-90.22"/>
      <circle cx="100" cy="100" r="80" stroke="#4f46e5" stroke-dasharray="51.55 502.65" stroke-dashoffset="-180.44"/>
      <circle cx="100" cy="100" r="80" stroke="#e11d48" stroke-dasharray="38.67 502.65" stroke-dashoffset="-231.99"/>
      <circle cx="100" cy="100" r="80" stroke="#d97706" stroke-dasharray="25.78 502.65" stroke-dashoffset="-270.66"/>
      <circle cx="100" cy="100" r="80" stroke="#ea580c" stroke-dasharray="38.67 502.65" stroke-dashoffset="-296.44"/>
      <circle cx="100" cy="100" r="80" stroke="#059669" stroke-dasharray="38.67 502.65" stroke-dashoffset="-335.11"/>
      <circle cx="100" cy="100" r="80" stroke="#0891b2" stroke-dasharray="25.78 502.65" stroke-dashoffset="-373.78"/>
      <circle cx="100" cy="100" r="80" stroke="#db2777" stroke-dasharray="12.89 502.65" stroke-dashoffset="-399.56"/>
      <circle cx="100" cy="100" r="80" stroke="#2563eb" stroke-dasharray="38.67 502.65" stroke-dashoffset="-412.45"/>
      <circle cx="100" cy="100" r="80" stroke="#16a34a" stroke-dasharray="51.55 502.65" stroke-dashoffset="-451.12"/>
    </svg>
    <div class="sk-donut__center">
      <span class="sk-donut__value">39+</span>
      <span class="sk-donut__label">algorithms</span>
    </div>
  </div>
  <ul class="sk-donut__legend">
    <li><span class="sk-donut__swatch" style="background:#0284c7"></span>Preprocessing<span class="sk-donut__count">5</span></li>
    <li><span class="sk-donut__swatch" style="background:#0d9488"></span>Imputation<span class="sk-donut__count">2</span></li>
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
| `SKStandardScaler` | Standardizes by removing the mean and scaling to unit variance | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKMinMaxScaler` | Scales each feature to a fixed range | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKRobustScaler` | Scaling robust to outliers through quartile statistics | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKOneHotEncoder` | One-hot encoding of categories | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKPolynomialFeatures` | Polynomial and interaction features | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Imputation · `sciencekit_impute`

| Estimator | Description | Status |
|---|---|---|
| Simple strategies | Mean, median, most frequent and constant | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKKNNImputer` | Multivariate imputation via nearest neighbors | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Linear models · `sciencekit_linear_model`

| Estimator | Description | Status |
|---|---|---|
| `SKLinearRegression` | Ordinary least squares | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKRidge` | Regression with L2 regularization | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKLasso` | Regression with L1 regularization (sparse support) | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKElasticNet` | Combined L1 + L2 regularization | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKLogisticRegression` | Regularized logistic classification | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKSGDClassifier` | Incremental classification via SGD · streaming (`SKLazySource`) | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKSGDRegressor` | Incremental regression via SGD · streaming (`SKLazySource`) | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Nearest neighbors · `sciencekit_neighbors`

| Structure/Estimator | Description | Status |
|---|---|---|
| `SKKNeighborsClassifier` | Classification by k nearest neighbors | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKKNeighborsRegressor` | Regression by k nearest neighbors | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKKDTree` | kd-tree for fast spatial search | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKBallTree` | Ball tree for arbitrary metrics | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Support vector machines · `sciencekit_svm`

| Estimator | Description | Status |
|---|---|---|
| `SKSVC` | Kernel-based support vector classification | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKSVR` | Support vector regression | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKLinearSVC` | Scalable linear SVM (dense and sparse) | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Trees and ensembles · `sciencekit_tree` + `sciencekit_ensemble`

| Estimator | Description | Status |
|---|---|---|
| `SKDecisionTreeClassifier` | Decision tree for classification | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKDecisionTreeRegressor` | Decision tree for regression | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKRandomForest` | Random forest aggregated in parallel with rayon | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKBagging` | Parallel bootstrap aggregation | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKGradientBoosting` | Sequential gradient boosting | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Unsupervised · `sciencekit_cluster`, `sciencekit_decomposition`, `sciencekit_outlier`

| Estimator | Description | Status |
|---|---|---|
| `SKKMeans` | K-means with O(1) random access via memmap | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKMiniBatchKMeans` | Mini-batch k-means with streaming (`SKLazySource`) | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKDBSCAN` | Density-based clustering | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKPCA` | Principal component analysis | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKTruncatedSVD` | Truncated SVD for sparse data | <span class="sk-pill sk-pill--neutral">Planned</span> |
| Anomaly detection | Outlier estimators from the scikit-learn family | <span class="sk-pill sk-pill--neutral">Planned</span> |

## Model selection, pipelines and metrics

| Item | Crate | Description | Status |
|---|---|---|---|
| `sk_train_test_split` | `sciencekit_model_selection` | Deterministic data partitioning | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKKFold` | `sciencekit_model_selection` | k-fold cross-validation | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKGridSearchCV` | `sciencekit_model_selection` | Grid search with cross-validation | <span class="sk-pill sk-pill--neutral">Planned</span> |
| `SKPipeline` | `sciencekit_pipeline` | Type-safe pipeline (associated types) and DAGs | <span class="sk-pill sk-pill--neutral">Planned</span> |
| Full metrics | `sciencekit_metrics` | accuracy, f1, MSE, confusion matrix and the remaining sklearn metrics | <span class="sk-pill sk-pill--neutral">Planned</span> |

<div class="sk-box sk-box--info">
<strong>Continuous acceleration:</strong> every completed algorithm receives, in separate changes,
a Python binding (<code>sciencekit_python</code>) and GPU backends (OpenCL → CUDA → ROCm) behind
<code>SKComputeBackend</code>. The CPU is the default backend, always present.
</div>

<div class="sk-btn-row">
  <a class="sk-btn sk-btn--primary" href="./architecture.html">Next: architecture →</a>
  <a class="sk-btn sk-btn--star" href="https://github.com/neurono-ml/sciencekit/issues">I want to implement one of these ⭐</a>
</div>
