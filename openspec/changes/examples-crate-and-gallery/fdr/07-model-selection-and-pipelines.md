# 07 — Model selection and pipelines

Model selection is where the corpus stops being about algorithms and starts being about
*method*: how the data is split, what is fitted inside the loop, and how the search
budget is spent.

---

### `cross_validated_score_of_a_classifier`

**Origin** — scikit-learn `examples/model_selection/plot_cross_val_score.py` · *"Cross-validation"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) |
| Feature selection | the two separating features |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` **inside** each fold |
| Pipeline | `SKStandardScaler → SKLogisticRegression`, evaluated by `SKKFold(5)` |
| Spectrum | the splitter arm (`SKKFold`, `SKStratifiedKFold`, `SKGroupKFold`); the fold count |
| Effect | the score of a single split is a noisy estimate of the model's performance; the cross-validated mean reduces that variance, and stratification keeps the class proportions constant across folds, which matters when the classes are imbalanced |
| Output | the per-fold score and its mean and standard deviation, and the score of the single-split arm for comparison |
| Availability | available now (`sciencekit_model_selection`) |

---

### `balancing_model_complexity_and_the_cross_validated_score`

**Origin** — scikit-learn `examples/model_selection/plot_grid_search_refit_callable.py` · *"Balance model complexity and cross-validated score"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 30, ten informative features |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRidge(alpha)`, searched by `SKGridSearchCV` |
| Spectrum | `alpha` over a logarithmic grid of 20 values; the scoring arm (R² against negative MSE) |
| Effect | the cross-validated score rises, peaks and falls across the grid: the peak is the best trade between fitting the data and generalizing; a grid too coarse may miss the peak, and a grid too fine multiplies the cost for no gain |
| Output | the mean score per candidate, the best candidate and its parameters, and the wall-clock time of the search |
| Availability | available now (`sciencekit_model_selection`) |

---

### `comparing_randomized_search_and_grid_search`

**Origin** — scikit-learn `examples/model_selection/plot_randomized_search.py` · *"Comparing randomized search and grid search for hyperparameter estimation"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 5000 × 20 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRidge(alpha)` and `SKGradientBoostingClassifier` searched by both strategies |
| Spectrum | the search strategy arm (grid vs randomized); the same evaluation budget for both |
| Effect | a grid spends its budget evenly over the *grid*, so a parameter that barely matters consumes as many evaluations as the important one; a randomized search samples the space, so with a small budget it finds the good region much more often — and it is the only practical option when a parameter is continuous |
| Output | the best score per strategy under the same budget, the distribution of the sampled points, and the wall-clock time of both |
| Availability | planned (`sciencekit_model_selection` — the randomized search) |

---

### `comparison_between_grid_search_and_successive_halving`

**Origin** — scikit-learn `examples/model_selection/plot_successive_halving_heatmap.py` · *"Comparison between grid search and successive halving"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 5000 × 20 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKGradientBoostingClassifier`, searched by `SKHalvingGridSearch` |
| Spectrum | the reduction factor; the initial number of candidates; the resource arm (number of iterations) |
| Effect | successive halving evaluates many candidates cheaply, keeps the best fraction and gives them more resources — it reaches a good candidate with far fewer total evaluations than a grid, at the price of a less thorough search of the promising region |
| Output | the score heatmap per resource level, the best candidate, and the total number of model fits of both strategies |
| Availability | available now (`sciencekit_model_selection` — the halving search) |

---

### `learning_curve_of_a_classifier`

**Origin** — scikit-learn `examples/model_selection/plot_learning_curve.html` · *"Learning curve"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 5000 × 20, a mixture of informative and noise features |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression` |
| Spectrum | the training-set size sweep; the classifier arm (logistic vs a tree) |
| Effect | the training score starts high and the validation score starts low; as the sample grows the two converge, and the *gap* between them is the overfitting that more data would remove — the gap, not the score, is what the curve is for |
| Output | the train and validation score against the sample size, the gap, and the extrapolation of the asymptote |
| Availability | planned (`sciencekit_model_selection` — the learning-curve utility) |

---

### `validation_curve_of_a_regularization_strength`

**Origin** — scikit-learn `examples/model_selection/plot_validation_curve.html` · *"Validation curve"*

| Stage | Recipe |
|---|---|
| Dataset | `load_diabetes` (embedded) |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRidge(alpha)` |
| Spectrum | the `alpha` sweep; the scoring arm |
| Effect | the validation score as a function of a single hyperparameter, with the spread across folds drawn around each point — the spread is what tells the reader whether the difference between two candidates is real |
| Output | the mean validation score and its standard deviation per `alpha`, and the argmax |
| Availability | planned (`sciencekit_model_selection` — the validation-curve utility) |

---

### `pipeline_of_scaling_and_a_linear_model`

**Origin** — scikit-learn `examples/compose/plot_pipeline_display.py` · *"Displaying Pipelines"* |

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 20 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKPipeline` — `SKStandardScaler` → `SKLogisticRegression` |
| Spectrum | the pipeline arm against the manual fit of the same two steps |
| Effect | the pipeline fits the scaler on the training data and applies it to the test data as one unit, so the two steps can never be accidentally unfitted or fitted on the whole dataset; the manual arm is the mistake the pipeline prevents |
| Output | the fitted stages, the predictions of both arms, their agreement, and the pipeline's structural description |
| Availability | available now (`sciencekit_pipeline`) |

---

### `column_transformer_over_mixed_column_types`

**Origin** — scikit-learn `examples/compose/plot_column_transformer.py` · *"Mixed-type columns in a ColumnTransformer"*

| Stage | Recipe |
|---|---|
| Dataset | a table with a numeric block, a categorical block and a text-like block |
| Feature selection | `SKColumnSelection` — the three blocks, by column name and by type |
| Feature engineering | the one-hot expansion of the categorical block |
| Preprocessing | `SKSimpleImputer` on the numeric block, `SKOneHotEncoder` on the categorical block |
| Pipeline | `SKColumnSelection` → per-block transforms → concatenation → `SKLogisticRegression` |
| Spectrum | the selection arm (by name vs by type); the remainder arm (drop vs passthrough) |
| Effect | the blocks need different treatments, and applying the wrong one is a silent error: a one-hot encoding of a numeric block explodes the width, while imputing a categorical block with the median invents a category that does not exist |
| Output | the transformed design's column names and width per arm, the fitted per-block transformers, and the model's score |
| Availability | available now (`sciencekit_pipeline`, `sciencekit_interop` column selection) |

---

### `feature_union_of_two_views_of_the_same_data`

**Origin** — scikit-learn `examples/compose/plot_feature_union.py` · *"Feature Union"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 20, of which five features carry the signal |
| Feature selection | none |
| Feature engineering | the second view: the polynomial expansion of the first five features |
| Preprocessing | `SKStandardScaler` on each branch |
| Pipeline | `SKFeatureUnion` — [raw scaled view ∥ polynomial view] → `SKRidge` |
| Spectrum | the union arm against the single-view arm; the weighting of the branches |
| Effect | the union lets a model see the raw features and their interactions at once; the width doubles, so the regularization has to cover twice as many coefficients, and the interaction terms are usually the ones that matter |
| Output | the width of the union, the score of both arms, and the coefficients of the interaction terms |
| Availability | available now (`sciencekit_pipeline`) |

---

### `streamed_pipeline_over_a_live_source`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"A pipeline that never holds the whole dataset"*

| Stage | Recipe |
|---|---|
| Dataset | a synthetic source delivering 100000 rows in blocks |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` with `partial_fit` |
| Pipeline | `SKStreamingPipeline` — `partial_fit(SKStandardScaler)` → `SKSGDClassifier` fitted on the transformed stream |
| Spectrum | the block size; the epoch count; the batch arm against a batch fit |
| Effect | the streamed model never holds more than one block in memory, and it tracks a drifting distribution; the batch model is more accurate on a stationary distribution but needs the whole data in memory |
| Output | the memory high-water mark of both arms, the score on a held-out final block, and the wall-clock time |
| Availability | available now (`sciencekit_pipeline` streaming contracts, `sciencekit_common` batching) |

---

### `custom_refit_strategy_of_a_grid_search`

**Origin** — scikit-learn `examples/model_selection/plot_grid_search_digits.py` · *"Custom refit strategy of a grid search with cross-validation"*

| Stage | Recipe |
|---|---|
| Dataset | `load_digits` (embedded) — 1797 × 64 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKKNeighborsClassifier`, searched by `SKGridSearchCV` |
| Spectrum | the refit strategy (the best single candidate vs the best candidate per fold) |
| Effect | refitting on the best candidate and refitting per fold are different objects: the first is one model trained on all the data, the second is one model per fold, which is what a deployment that must predict per data batch needs |
| Output | the best parameters, the refitted model's score, and the per-fold models under the second strategy |
| Availability | planned (`sciencekit_model_selection` — the per-fold refit strategy) |
