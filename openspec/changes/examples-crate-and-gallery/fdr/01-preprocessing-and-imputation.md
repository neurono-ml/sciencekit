# 01 — Preprocessing and imputation

The preprocessing family is sciencekit's most mature surface: four scalers, the one-hot
encoder and the polynomial expansion, plus the imputation crate. These examples are the
*first* wave of the corpus and the ground material for the book's chapters on scaling
and on missing data.

---

### `standard_scaler_on_a_mixed_scale_dataset`

**Origin** — scikit-learn `examples/preprocessing/plot_scaling.py` · *"Standardize a feature by removing the mean and scaling to unit variance"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 500 × 4, two informative features deliberately on different scales (one in the thousands, one in the tenths) |
| Feature selection | none — the four synthetic features |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — `with_mean(true)`, `with_std(true)` |
| Pipeline | `SKStandardScaler → SKLinearRegression` |
| Spectrum | scaler **off** vs **on**; and `with_std(false)` as the "center only" arm |
| Effect | with the scaler off, the large-scale feature dominates the least-squares solution and the small-scale feature's coefficient collapses toward zero; centering alone recovers the coefficient magnitudes but not the conditioning |
| Output | per-feature mean and standard deviation before and after; the coefficient vector of the three arms |
| Availability | available now (`sciencekit_preprocessing`, `sciencekit_linear_model`) |

---

### `minmax_scaler_to_a_bounded_range`

**Origin** — scikit-learn `examples/preprocessing/plot_scaling.py` · *"MinMaxScaler"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) — 150 × 4, features on incomparable scales (cm versus fraction) |
| Feature selection | `sepal length` and `petal length` — the two features with the widest dynamic range |
| Feature engineering | none |
| Preprocessing | `SKMinMaxScaler` — `feature_range((0, 1))`, `clip(true)` |
| Pipeline | `SKMinMaxScaler → SKLinearRegression` |
| Spectrum | `feature_range ∈ {(0,1), (-1,1), (2,3)}`; `clip` on and off |
| Effect | the mapped range is exactly the configured one; with `clip` off, out-of-range test samples land outside it, which changes the predictions of the fitted model |
| Output | the per-feature minimum and maximum after the transform; the fraction of test values outside the training range |
| Availability | available now (`sciencekit_preprocessing`) |

---

### `robust_scaler_against_outliers`

**Origin** — scikit-learn `examples/preprocessing/plot_scaling.py` · *"RobustScaler"*

| Stage | Recipe |
|---|---|
| Dataset | `load_diabetes` (embedded) with 5 % of every feature's entries replaced by outliers at ±10 standard deviations |
| Feature selection | none |
| Feature engineering | the outliers are injected, not derived |
| Preprocessing | `SKRobustScaler` — `quantile_range((25.0, 75.0))` (the interquartile range), `center(true)`, `scale(true)` |
| Pipeline | `SKRobustScaler → SKRidge(alpha = 1.0)` |
| Spectrum | `SKStandardScaler` vs `SKRobustScaler` on the same contaminated data; `quantile_range ∈ {(25,75), (5,95)}` |
| Effect | the standard scaler's mean and variance are dragged by the outliers, so every clean point is pushed off-centre; the robust scaler's median and interquartile range are not, and the ridge coefficients stay close to the clean-data fit |
| Output | the scaler's fitted center and scale per feature under both scalers; the ridge coefficients |
| Availability | available now (`sciencekit_preprocessing`) |

---

### `one_hot_encoding_of_categorical_features`

**Origin** — scikit-learn `examples/preprocessing/plot_one_hot_encoder.py` · *"One-hot encoding of categorical features"*

| Stage | Recipe |
|---|---|
| Dataset | `fetch_openml`-style tabular data replaced by the embedded adult-census-like table: 6 categorical columns of low cardinality |
| Feature selection | the six categorical columns; the numeric columns are excluded from the encoding and kept as-is |
| Feature engineering | none |
| Preprocessing | `SKOneHotEncoder` — `handle_unknown(ignore)` |
| Pipeline | `SKOneHotEncoder` → dense design → `SKLogisticRegression` |
| Spectrum | `handle_unknown ∈ {ignore, error}`; dense output vs sparse output |
| Effect | with `handle_unknown(error)`, a category unseen in training aborts the fit; with `ignore`, it is encoded as an all-zero row and the prediction still succeeds |
| Output | the encoded column names, the number of columns before and after, and the predicted class for a sample carrying an unseen category |
| Availability | available now (`sciencekit_preprocessing`) |

---

### `polynomial_feature_expansion_of_interactions`

**Origin** — scikit-learn `examples/preprocessing/plot_polynomial_features.py` · *"Polynomial expansion"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 200 × 3 with three interacting ground-truth terms |
| Feature selection | none |
| Feature engineering | the expansion itself: powers of each feature and their pairwise products |
| Preprocessing | `SKPolynomialFeatures` — `degree(2)`, `interaction_only(false)`, `include_bias(false)` |
| Pipeline | `SKPolynomialFeatures → SKLinearRegression` |
| Spectrum | `degree ∈ {1, 2, 3}`; `interaction_only ∈ {false, true}` |
| Effect | at `degree = 1` the model cannot represent the interaction and underfits badly; at `degree = 2` the interaction term is present in the design and the fit becomes nearly exact; `interaction_only` keeps the pure powers but drops the products |
| Output | the number of output columns per setting, the coefficients of the three ground-truth interaction terms, and the R² |
| Availability | available now (`sciencekit_preprocessing`) |

---

### `imputing_missing_values_before_building_an_estimator`

**Origin** — scikit-learn `examples/impute/plot_missing_values.py` · *"Imputing missing values before building an estimator"*

| Stage | Recipe |
|---|---|
| Dataset | `load_diabetes` (embedded) with 10 % of the entries set to missing, introduced by a recorded mechanism (masking at random) |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKSimpleImputer` — `missing_values(NaN)`, `fill_value(median)`, `keep_empty_features(true)` |
| Pipeline | `SKSimpleImputer → SKStandardScaler → SKLinearRegression` |
| Spectrum | strategy `mean` vs `median` vs `most_frequent`; and the "drop the incomplete rows" arm as the reference |
| Effect | dropping the incomplete rows discards 10 % of the sample and changes the fitted coefficients; each imputation strategy keeps every row but biases the affected features differently, and the median is the most resistant to the skewed distributions in the diabetes data |
| Output | the number of rows kept, the imputed values of a few masked entries, and the test R² of the three arms |
| Availability | available now (`sciencekit_impute`, `sciencekit_linear_model`) |

---

### `missing_indicator_as_an_explicit_signal`

**Origin** — scikit-learn `examples/impute/plot_missing_values.py` · *"Missing-indicator" arm of the imputation example"*

| Stage | Recipe |
|---|---|
| Dataset | the same masked diabetes design |
| Feature selection | none |
| Feature engineering | the indicator columns are derived features: one binary column per feature that ever had a missing value |
| Preprocessing | `SKMissingIndicator` — `features(only-missing)` |
| Pipeline | `SKSimpleImputer → SKMissingIndicator → SKStandardScaler → SKLinearRegression` |
| Spectrum | indicator absent vs present; `features ∈ {all, only-missing}` |
| Effect | with the indicator present, the model can learn that a feature was *missing* rather than that it was *small*, which is a different signal entirely; on a dataset where missingness correlates with the target, this raises the achievable score |
| Output | the indicator matrix, the number of added columns, and the test score of both arms |
| Availability | available now (`sciencekit_impute`) |

---

### `knn_imputation_of_missing_feature_values`

**Origin** — scikit-learn `examples/impute/plot_missing_values.py` · *"Nearest-neighbor imputation"*

| Stage | Recipe |
|---|---|
| Dataset | `load_diabetes` (embedded) with 10 % of the entries masked |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKKNNImputer` — `nearest_neighbors_count(5)`, distance-weighted over the observed entries |
| Pipeline | `SKKNNImputer → SKStandardScaler → SKLinearRegression` |
| Spectrum | `nearest_neighbors_count ∈ {1, 3, 5, 10}`; `SKSimpleImputer` as the reference |
| Effect | a single neighbour copies the nearest row's value and propagates its noise; more neighbours interpolate and reduce the variance of the imputed values, which lowers the test error until the neighbours start spanning different clusters |
| Output | the imputed values under each neighbour count, their variance, and the test R² |
| Availability | available now (`sciencekit_impute`, `sciencekit_neighbors`) |

---

### `scaling_inside_a_cross_validated_pipeline`

**Origin** — scikit-learn `examples/preprocessing/plot_scaling.py` · *"Scaling inside a cross-validated pipeline"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 20, ten informative features, ten noise features |
| Feature selection | the ten informative features are *selected by name* only in the didactic sense — the estimator itself receives all twenty, and the informative subset is used to interpret the coefficients |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` inside the cross-validation loop |
| Pipeline | `SKStandardScaler → SKLogisticRegression` wrapped in `SKKFold(5)` |
| Spectrum | scaler fitted **inside** each fold vs fitted **once** on the whole dataset |
| Effect | fitting the scaler once leaks the test fold's mean and variance into training, and the reported cross-validated score becomes optimistically biased; the difference grows with the fold count and with the data's non-stationarity |
| Output | the per-fold scores under both regimes and their mean |
| Availability | available now (`sciencekit_model_selection`, `sciencekit_preprocessing`) |

---

### `streaming_scaler_under_a_shifting_distribution`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Incremental standardization under a drifting distribution"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` generated in two blocks with a deliberately shifted second block (the mean of every feature moves by three standard deviations) |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` with `partial_fit` over the stream |
| Pipeline | stream → `partial_fit(SKStandardScaler)` → `SKLinearRegression` fitted on the transformed stream |
| Spectrum | one-shot fit on the whole stream vs incremental `partial_fit` in blocks of 50 |
| Effect | the one-shot scaler centres on the average of both blocks and leaves each block off-centre; the incremental scaler ends centred on the *latest* block, so the model's coefficients adapt to the current regime |
| Output | the fitted mean per block, the coefficient drift, and the residual error on the second block |
| Availability | available now (`sciencekit_preprocessing`, `sciencekit_common` streaming contracts) |

---

### `preprocessing_chain_cheat_sheet`

**Origin** — *no scikit-learn origin — a sciencekit-native reference page* · *"The preprocessing chain, end to end"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` with a masked column and two categorical columns injected |
| Feature selection | `SKColumnSelection` — the numeric block, the categorical block, the column with missing values |
| Feature engineering | the polynomial expansion of the numeric block |
| Preprocessing | `SKSimpleImputer` → `SKOneHotEncoder` → `SKStandardScaler` → `SKPolynomialFeatures` |
| Pipeline | the full chain, as a `SKPipeline` with typed stages |
| Spectrum | each stage enabled and disabled in turn |
| Effect | removing the imputer aborts the fit on the missing column; removing the scaler changes the conditioning of the linear model; removing the encoder makes the categorical columns uninterpretable to the estimator |
| Output | the design shape at each step of the chain, the number of columns, and the model's test score with each stage removed |
| Availability | available now (`sciencekit_pipeline`, `sciencekit_impute`, `sciencekit_preprocessing`) |
