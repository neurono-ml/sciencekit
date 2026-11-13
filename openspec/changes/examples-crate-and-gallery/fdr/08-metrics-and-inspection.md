# 08 — Metrics and inspection

The metrics crate is complete enough to carry a family of its own. These examples answer
the question a practitioner actually asks: *which number should I look at, and what
does it hide?*

---

### `confusion_matrix_of_a_classifier`

**Origin** — scikit-learn `examples/model_selection/plot_confusion_matrix.py` · *"Evaluate the performance of a classifier with Confusion Matrix"*

| Stage | Recipe |
|---|---|
| Dataset | `load_digits` (embedded), restricted to three classes |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKKNeighborsClassifier` |
| Spectrum | the classifier arm; the confusion matrix's normalization (absolute, per row, per column) |
| Effect | the matrix shows the two error directions separately, which accuracy hides: a model that never predicts the minority class can look excellent on a rare positive; the row normalization turns each row into "what did the model do with the true class", the column into "where did the predictions of this class come from" |
| Output | the raw and the two normalized matrices, accuracy, and the per-class precision and recall |
| Availability | available now (`sciencekit_metrics`, `sciencekit_neighbors`) |

---

### `roc_curve_of_a_binary_classifier`

**Origin** — scikit-learn `examples/model_selection/plot_roc_curve.html` · *"Receiver Operating Characteristic (ROC)"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 2000 × 20 with 10 % positives |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression` |
| Spectrum | the classifier arm; the threshold sweep that traces the curve |
| Effect | the ROC curve is the trade-off between the true-positive rate and the false-positive rate as the threshold moves; it is invariant to the class proportions, which is exactly why it must be read together with the prevalence when the classes are imbalanced |
| Output | the curve, the area under it, the operating point at a chosen threshold, and the score at the default threshold |
| Availability | available now (`sciencekit_metrics` ranking and probability scorers) |

---

### `precision_recall_curve_under_class_imbalance`

**Origin** — scikit-learn `examples/model_selection/plot_precision_recall_curve.html` · *"Precision-Recall curve"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 2000 × 20 with 1 % positives |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression` |
| Spectrum | the threshold sweep; the classifier arm |
| Effect | with a rare positive class the precision falls as the recall rises, because each additional true positive is paid for with many false positives; the area under the precision–recall curve depends on the prevalence, so it cannot be compared across datasets with different positive rates |
| Output | the curve, the average precision, the precision at the chosen operating point, and the number of false positives behind it |
| Availability | available now (`sciencekit_metrics` ranking scorers) |

---

### `detection_error_tradeoff_curve`

**Origin** — scikit-learn `examples/model_selection/plot_det.html` · *"Detection error tradeoff (DET) curve"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 5000 × 20, three difficulty levels |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression` |
| Spectrum | the threshold sweep, plotted on the normal-deviation scale |
| Effect | the DET curve plots the false-negative rate against the false-positive rate on axes scaled by the normal quantile function, which turns the region of interest into a straight line; the distance between two curves is the discriminability of the two systems, independent of the operating point |
| Output | the DET curves on the normal-deviation scale, the equal-error-rate point, and the distance between the curves |
| Availability | planned (`sciencekit_metrics` — the DET utility on top of the existing ranking scorers) |

---

### `multi_metric_evaluation_of_a_cross_validated_score`

**Origin** — scikit-learn `examples/model_selection/plot_multi_metric_evaluation.py` · *"Demonstration of multi-metric evaluation on cross_val_score and GridSearchCV"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 20 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression`, evaluated by `SKKFold(5)` |
| Spectrum | the metric arm (accuracy, balanced accuracy, F1, AUC, average precision) |
| Effect | the metrics disagree on purpose: accuracy rewards the majority class, balanced accuracy weights the classes equally, F1 ignores the true negatives, and the AUC measures the *ranking* rather than the decisions — a model that ranks well can still be badly calibrated at the default threshold |
| Output | the per-fold value of each metric, their disagreement on the same model, and the metric that best separates two candidate models |
| Availability | available now (`sciencekit_metrics` scorers, `sciencekit_model_selection`) |

---

### `class_likelihood_ratios_for_a_classifier`

**Origin** — scikit-learn `examples/model_selection/plot_likelihood_ratios.html` · *"Class Likelihood Ratios to measure classification performance"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 2000 × 20 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression` |
| Spectrum | the threshold sweep, from which the likelihood ratios are computed |
| Effect | the positive likelihood ratio is how much more often the score occurs in the positive class than in the negative class; it tells the reader how much the evidence moves, independently of the prevalence, which is what makes it usable across populations |
| Output | the likelihood ratios against the threshold, the pre-test and post-test probabilities, and the number needed to treat |
| Availability | planned (`sciencekit_metrics` — the likelihood-ratio utility) |

---

### `permutation_importance_against_forest_importance`

**Origin** — scikit-learn `examples/inspection/plot_permutation_importance.py` · *"Permutation Importance vs Random Forest Feature Importance (MDI)"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 20, ten informative features, ten noise features |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRandomForestClassifier` |
| Spectrum | the importance arm (impurity-based against permutation-based); the number of repetitions |
| Effect | the impurity importance is measured on the training data and is biased towards features with many distinct values; the permutation importance shuffles a feature in the *test* data and measures the drop in score, which is a statement about the model and generalizes; the two rankings disagree exactly on the high-cardinality noise features |
| Output | both importance vectors with their dispersion, the ranking of the ten informative features, and the number of features the two methods disagree on |
| Availability | planned (`sciencekit_metrics` — the permutation importance) |

---

### `permutation_importance_with_correlated_features`

**Origin** — scikit-learn `examples/inspection/plot_permutation_importance_multicollinear.py` · *"Permutation Importance with Multicollinear or Correlated Features"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` with three duplicated informative features, each a copy of the others |
| Feature selection | none |
| Feature engineering | the duplication itself |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRandomForestClassifier` |
| Spectrum | the importance arm; the arm with the duplicated features removed |
| Effect | when features are interchangeable, permuting any one of them leaves the model's performance almost unchanged, because the others still carry the information — the importance is split arbitrarily between them and the total drops when a duplicate is removed; a correlated feature can therefore hide a genuinely useless one |
| Output | the importances of the duplicated block, the total importance of the block before and after the removal, and the score change |
| Availability | planned (`sciencekit_metrics` — the permutation importance) |

---

### `partial_dependence_of_a_model`

**Origin** — scikit-learn `examples/inspection/plot_partial_dependence.py` · *"Partial Dependence and Individual Conditional Expectation Plots"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` with a non-linear ground truth in two interacting features |
| Feature selection | the two interacting features |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKGradientBoostingRegressor` |
| Spectrum | the feature arm; the interaction arm; the grid resolution |
| Effect | the partial dependence averages the model's prediction over the other features, which shows the *marginal* effect the model has learned; the individual conditional expectation plot shows the prediction for each individual sample, which exposes the heterogeneity the average hides |
| Output | the partial dependence curve per feature, the individual curves, and the spread around the average |
| Availability | planned (`sciencekit_metrics` — the partial-dependence utility) |

---

### `probability_calibration_curve_of_a_classifier`

**Origin** — scikit-learn `examples/calibration/plot_calibration_curve.py` · *"Probability Calibration curves"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 5000 × 20, deliberately miscalibrated by overfitting a flexible model |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKGradientBoostingClassifier` |
| Spectrum | the calibration arm (none, sigmoid, isotonic); the sample count per bin |
| Effect | a model can rank perfectly and still be badly calibrated: the predicted probability of 0.8 occurs only 60 % of the time; the calibration curve shows the gap, and the sigmoid correction restores the probabilities at the price of a small loss in ranking |
| Output | the reliability curve against the diagonal, the calibration error, and the log loss before and after |
| Availability | planned (`sciencekit_metrics` — the calibration metrics are planned with the calibration and multilabel metrics change) |

---

### `classification_report_of_a_multiclass_model`

**Origin** — scikit-learn `examples/classification/plot_classification_probability.py` · *"Plot classification probability"*

| Stage | Recipe |
|---|---|
| Dataset | `load_digits` (embedded) |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKKNeighborsClassifier` |
| Spectrum | the number of neighbours; the prediction arm (hard label against the probability vector) |
| Effect | the probability vector of a neighbour classifier is the fraction of the neighbours that voted for each class, so its confidence is a direct count of evidence; the hardest digits are those whose neighbours disagree |
| Output | the per-class precision, recall and F1, the accuracy, and the probability vectors of the least and most confident samples |
| Availability | available now (`sciencekit_metrics`, `sciencekit_neighbors`) |
