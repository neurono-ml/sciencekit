# 09 — Feature selection and feature engineering

Two ways of shrinking the input space. **Feature selection** keeps a subset of the
original columns; **feature engineering** builds new ones. The examples are paired so
the reader can compare *what is kept* with *what is derived*.

---

### `univariate_feature_selection_by_a_statistical_test`

**Origin** — scikit-learn `examples/feature_selection/plot_feature_selection.py` · *"Univariate Feature Selection"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 50, ten informative features, the rest noise |
| Feature selection | a univariate test per feature, keeping the `k` best by score: the F-test and the mutual-information score |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` (the F-test needs the features to be comparable) |
| Pipeline | `SKStandardScaler → univariate selection(k) → SKLogisticRegression` |
| Spectrum | `k ∈ {5, 10, 20, 50}`; the test arm (F-test against mutual information) |
| Effect | the F-test assumes linearity and is fooled by a purely non-linear relationship; the mutual information detects any dependency but needs many samples per feature; both are *univariate*, so a feature that is useless alone and decisive in combination is discarded by both |
| Output | the selected feature indices per `k` and per test, the overlap of the two selections, and the score of each arm |
| Availability | planned (`sciencekit_feature_selection` — the family is not implemented yet) |

---

### `recursive_feature_elimination`

**Origin** — scikit-learn `examples/feature_selection/plot_rfe_digits.py` · *"Recursive Feature Elimination"*

| Stage | Recipe |
|---|---|
| Dataset | `load_digits` (embedded) — 1797 × 64 |
| Feature selection | recursive elimination: fit the model, drop the least important feature, refit, repeat |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → RFE(estimator, number_of_features) → SKLogisticRegression` |
| Spectrum | `number_of_features ∈ {10, 20, 40, 64}`; the estimator arm used to rank the features |
| Effect | elimination is *model-driven*: it removes the feature whose removal hurts the model least, which for a linear model means the most collinear one — so RFE on a linear model cannot remove two features that are individually useless but jointly decisive, and it will happily drop one of a correlated pair |
| Output | the surviving feature indices per count, the score per count, and the features dropped at each step |
| Availability | planned (`sciencekit_feature_selection`) |

---

### `model_based_and_sequential_feature_selection`

**Origin** — scikit-learn `examples/feature_selection/plot_select_from_model_diabetes.py` · *"Model-based and Sequential Feature Selection"*

| Stage | Recipe |
|---|---|
| Dataset | `load_diabetes` (embedded) — 442 × 10 |
| Feature selection | the model-based arm (keep the features with the largest coefficients) and the sequential arm (forward or backward stepwise) |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — the coefficients of an unstandardized model are not comparable across features |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → selection → SKRidge` |
| Spectrum | the number of selected features ∈ {2, 4, 6, 10}; the selection arm |
| Effect | the model-based selection ranks the features by the magnitude of their coefficient, which is meaningless when the features are correlated — the pair shares the credit arbitrarily; the sequential selection evaluates the *model* after each addition, which is slower but sees the effect of a combination |
| Output | the selected features per count per arm, the score of each arm, and the coefficient vector of the selected subset |
| Availability | planned (`sciencekit_feature_selection`) |

---

### `feature_selection_inside_a_pipeline`

**Origin** — scikit-learn `examples/feature_selection/plot_feature_selection_pipeline.py` · *"Pipeline ANOVA SVM"*

| Stage | Recipe |
|---|---|
| Dataset | `load_digits` (embedded) — 1797 × 64 |
| Feature selection | the univariate selector, inside the cross-validation loop |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → selection(k) → SKSVC`, evaluated by `SKKFold(5)` |
| Spectrum | `k ∈ {5, 10, 20, 40}` |
| Effect | selecting the features on the whole dataset before the split leaks the test fold into the selection and inflates the score; inside the pipeline each fold selects on its own training part, and the score drops to the honest value |
| Output | the cross-validated score per `k` with the selection inside and outside the loop, and the gap between them |
| Availability | planned (`sciencekit_feature_selection`, `sciencekit_model_selection`) |

---

### `feature_agglomeration_against_univariate_selection`

**Origin** — scikit-learn `examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py` · *"Feature agglomeration vs. univariate selection"*

| Stage | Recipe |
|---|---|
| Dataset | `load_digits` (embedded) |
| Feature selection | feature agglomeration — merge the features into groups by similarity — against the univariate selector |
| Feature engineering | the group means become the new features |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → agglomeration(number_of_clusters) → SKLogisticRegression` |
| Spectrum | the number of feature clusters ∈ {5, 10, 20, 40} |
| Effect | agglomeration groups *similar* features and keeps one representative per group, which captures the redundancy of a correlated block; the univariate selector keeps the individually strongest features and may drop a whole group whose members are each weak alone — the two methods fail in opposite directions |
| Output | the group memberships, the representative per group, the score of both methods, and the overlap of the retained features |
| Availability | planned (`sciencekit_feature_selection`) |

---

### `cyclic_encoding_of_a_time_feature`

**Origin** — scikit-learn `examples/applications/plot_cyclical_feature_engineering.py` · *"Time-related feature engineering"*

| Stage | Recipe |
|---|---|
| Dataset | a synthetic time series with a daily and a weekly cycle |
| Feature selection | the timestamp column |
| Feature engineering | the *cyclic* encoding of the hour: `sin(2π hour / 24)` and `cos(2π hour / 24)`; the same for the day of the week |
| Preprocessing | `SKStandardScaler` |
| Pipeline | timestamp → cyclic encoding → `SKGradientBoostingRegressor` |
| Spectrum | the encoding arm (raw hour, ordinal hour, cyclic encoding); the model arm |
| Effect | the ordinal hour tells the model that 23 and 0 are far apart and that 0 and 1 are neighbours; the cyclic encoding puts every hour at the correct distance from every other, so midnight and one o'clock are close while midnight and noon are far — the distance structure is the whole point |
| Output | the two derived columns for a day, the score of the three arms, and the model's learned shape over the day |
| Availability | planned (`sciencekit_examples` — the encoding is an example-local feature-engineering step, not a library transform) |

---

### `lagged_features_for_a_time_series`

**Origin** — scikit-learn `examples/applications/plot_time_series_lagged_features.py` · *"Lagged features for time series forecasting"*

| Stage | Recipe |
|---|---|
| Dataset | a synthetic seasonal time series, 1000 points |
| Feature selection | the lagged columns chosen by the example (the lags themselves) |
| Feature engineering | the lag construction: `y(t-1)`, `y(t-2)`, … as additional columns; plus the calendar features |
| Preprocessing | `SKStandardScaler` |
| Pipeline | lag construction → `SKStandardScaler` → `SKGradientBoostingRegressor` |
| Spectrum | the number of lags ∈ {1, 3, 7, 14}; the model arm (linear against boosting) |
| Effect | the number of lags is the memory of the model: one lag is enough for a strong trend, a seasonal series needs as many lags as the season; too few lags underfit the season, too many let the model chase the noise of the far past |
| Output | the score per lag count, the fitted curve against the series, and the residual autocorrelation |
| Availability | planned (`sciencekit_examples` — the lag construction is example-local) |

---

### `one_hot_encoding_of_a_low_cardinality_feature`

**Origin** — scikit-learn `examples/preprocessing/plot_one_hot_encoder.py` · *"One-hot encoding of categorical features"* |

| Stage | Recipe |
|---|---|
| Dataset | a table with one low-cardinality categorical column (four levels) and one high-cardinality column (one level per row) |
| Feature selection | the categorical column |
| Feature engineering | the one-hot expansion |
| Preprocessing | `SKOneHotEncoder` — the arm with and without the drop of the first level |
| Pipeline | `SKOneHotEncoder → SKLogisticRegression` |
| Spectrum | the number of levels ∈ {2, 4, 20, 200}; the drop-first arm |
| Effect | a categorical feature with two hundred levels becomes two hundred columns, most of them nearly empty, and the model can fit a coefficient to each — with few samples per level that is pure overfitting; dropping the first level removes the redundancy at the price of an asymmetric coding, and the coefficients become interpretable as contrasts |
| Output | the width of the encoded design per level count, the number of empty columns, and the score under the drop-first arm |
| Availability | available now (`sciencekit_preprocessing`) |

---

### `interaction_features_between_categorical_and_numeric_columns`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Cross-type interactions"*

| Stage | Recipe |
|---|---|
| Dataset | a table with a numeric block and a categorical block whose levels modify the numeric effect |
| Feature selection | `SKColumnSelection` — the numeric block and the categorical block |
| Feature engineering | the product of each numeric feature with the one-hot encoding of the categorical feature |
| Preprocessing | `SKSimpleImputer` → `SKOneHotEncoder` → `SKStandardScaler` |
| Pipeline | the interaction construction → `SKGradientBoostingRegressor` |
| Spectrum | the interaction arm (present against absent) |
| Effect | a tree model can learn the interaction by itself, but a linear model cannot: without the interaction columns the categorical level shifts the *intercept* only, while the true effect is a change of *slope*; the example shows the linear model failing where the boosted model succeeds, and the interaction columns fixing the linear one |
| Output | the score of both models with and without the interaction columns, and the recovered slope per category |
| Availability | planned (`sciencekit_examples` — the interaction construction is example-local) |

---

### `target_encoding_of_a_high_cardinality_feature`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Target encoding, and the leakage it invites"*

| Stage | Recipe |
|---|---|
| Dataset | a table with a high-cardinality categorical column (one level per fifty rows) and a binary target that depends on the level |
| Feature selection | the categorical column |
| Feature engineering | the *target* encoding: each level is replaced by the mean target of its training rows |
| Preprocessing | the encoding is computed **inside** the cross-validation loop |
| Pipeline | target encoding → `SKLogisticRegression`, evaluated by `SKKFold(5)` |
| Spectrum | the encoding arm (target encoding against one-hot); the smoothing arm (with and without the prior blend) |
| Effect | target encoding replaces a hundred columns with one and captures the level's effect directly, but computing it on the whole dataset leaks the target of the test rows into the features — the encoded value of a test row contains its own label; inside the loop the encoding is honest, and the honest score is the lower one |
| Output | the width of the design per arm, the cross-validated score with the encoding inside and outside the loop, and the gap between them |
| Availability | planned (`sciencekit_examples` — the encoding is example-local, and the leakage demonstration is its point) |

---

### `variance_threshold_as_a_zero_variance_filter`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Dropping the constants"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` with three constant columns added |
| Feature selection | the zero-variance filter, applied before anything else |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — a constant column survives the scaler as a column of zeros |
| Pipeline | `SKColumnSelection` (drop the constants) → `SKStandardScaler` → `SKLogisticRegression` |
| Spectrum | the filter arm; the position of the filter in the chain |
| Effect | a constant column carries no information but does carry a coefficient, and with a regularization path it can absorb penalty budget; the filter is the cheapest possible selection and the safest place to start |
| Output | the number of dropped columns, the design width before and after, and the coefficient vector showing the empty dimensions |
| Availability | planned (`sciencekit_examples` — the filter is a column selection over the fitted design) |
