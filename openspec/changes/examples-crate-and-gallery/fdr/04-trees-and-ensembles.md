# 04 — Trees and ensembles

Three ideas in one family: a single tree splits greedily and overfits; bagging averages
many trees to cancel that variance; boosting adds trees that correct the previous ones.
The examples are arranged so the reader sees the three behaviours side by side.

---

### `decision_surface_of_a_decision_tree`

**Origin** — scikit-learn `examples/tree/plot_iris_dtc.py` · *"Decision surface of decision trees trained on the iris dataset"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) — 150 × 4, three classes |
| Feature selection | the two features that separate the classes: sepal length and petal length |
| Feature engineering | none |
| Preprocessing | none — a tree is invariant to monotone rescaling of the features, which the example states and asserts |
| Pipeline | `SKDecisionTreeClassifier` |
| Spectrum | `maximum_depth ∈ {1, 2, 3, None}`; the criterion arm |
| Effect | depth 1 gives a single split and can only separate one class; every added level carves one more axis-aligned rectangle, and the boundary becomes a staircase that hugs the training points; at full depth the tree memorizes the sample and the staircase becomes jagged |
| Output | the decision surface, the split thresholds, the leaf sample counts, the feature importances and the test accuracy per depth |
| Availability | available now (`sciencekit_tree`) |

---

### `understanding_the_structure_of_a_decision_tree`

**Origin** — scikit-learn `examples/tree/plot_unveil_tree_structure.py` · *"Understanding the decision tree structure"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) |
| Feature selection | all four features |
| Feature engineering | none |
| Preprocessing | none |
| Pipeline | `SKDecisionTreeClassifier` |
| Spectrum | `maximum_depth ∈ {1, 2, 3}`; `minimum_samples_split ∈ {2, 10, 40}` |
| Effect | each node splits on the feature that most reduces the impurity; the impurity of a node is the weighted sum of its children's impurities; `minimum_samples_split` stops the growth before a node becomes a single sample |
| Output | the tree structure as text (nodes, features, thresholds, sample counts, impurity), the feature importances, and the depth of the fitted tree |
| Availability | available now (`sciencekit_tree`) |

---

### `post_pruning_a_decision_tree_by_cost_complexity`

**Origin** — scikit-learn `examples/tree/plot_cost_complexity_pruning.py` · *"Post pruning decision trees with cost complexity pruning"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) with 10 % of the labels replaced by noise, so the tree overfits |
| Feature selection | all four features |
| Feature engineering | the label noise is injected, not derived |
| Preprocessing | none |
| Pipeline | `SKDecisionTreeClassifier` grown to full depth, then pruned at a cost-complexity threshold |
| Spectrum | the pruning threshold `alpha ∈ {0.0, 0.001, 0.01, 0.05, 0.1}` |
| Effect | the fully grown tree has many leaves and a perfect training score; each pruning step removes a subtree whose impurity gain no longer pays for its complexity, and the test error stops rising while the tree shrinks |
| Output | the number of leaves and the depth per threshold, the training and test error per threshold, and the effective alpha |
| Availability | planned (`sciencekit_tree` — the cost-complexity pruning on top of the implemented tree) |

---

### `random_forest_on_the_iris_dataset`

**Origin** — scikit-learn `examples/ensemble/plot_forest_iris.py` · *"Decision surfaces of ensembles of trees on the iris dataset"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) |
| Feature selection | the two separating features |
| Feature engineering | none |
| Preprocessing | none |
| Pipeline | `SKRandomForestClassifier` |
| Spectrum | `number_of_estimators ∈ {1, 5, 50}`; `maximum_features ∈ {sqrt, all}`; `bootstrap` on and off |
| Effect | a single tree draws an axis-aligned staircase that follows the noise; averaging many trees drawn on bootstrap samples produces a smooth, robust boundary; with `maximum_features = all` every tree sees every feature and the diversity that makes the forest work disappears |
| Output | the decision surface per number of estimators, the out-of-bag score, the feature importances with their variance across trees, and the test accuracy |
| Availability | available now (`sciencekit_ensemble`) |

---

### `out_of_bag_errors_for_random_forests`

**Origin** — scikit-learn `examples/ensemble/plot_ensemble_oob.py` · *"OOB Errors for Random Forests"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 20, ten informative features |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRandomForestClassifier(oob)` |
| Spectrum | `number_of_estimators ∈ {10, 50, 200}`; `bootstrap` on (required for the OOB estimate) |
| Effect | every sample is left out of roughly a third of the trees, so the votes of those trees form a free validation set; the OOB error converges towards the cross-validated error as the forest grows, which makes the OOB estimate trustworthy without a separate split |
| Output | the OOB error against the number of estimators, the training error, and the gap between them |
| Availability | available now (`sciencekit_ensemble`) |

---

### `feature_importances_of_a_forest_of_trees`

**Origin** — scikit-learn `examples/ensemble/plot_forest_importances.py` · *"Feature importances with a forest of trees"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 20, ten informative features, ten noise features |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRandomForestClassifier` |
| Spectrum | the importances arm (impurity-based vs permutation-based); `maximum_features ∈ {sqrt, all}` |
| Effect | the impurity-based importance is biased towards high-cardinality continuous features and is computed on the training data, so it can flatter a useless feature; the permutation importance measures the drop in score when a feature is shuffled, which is a statement about the model, not the data |
| Output | the importances per feature with the forest's standard deviation, the ranking of the ten informative features, and the two importance measures side by side |
| Availability | available now (`sciencekit_ensemble` for the impurity importances; the permutation arm is planned in `sciencekit_metrics`) |

---

### `bagging_a_base_estimator_on_bootstrap_samples`

**Origin** — scikit-learn `examples/ensemble/plot_bagging_classifier.py` · *"Bagging classifier"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 20 with two informative features drawn on a circle, so a single tree needs a staircase |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKBaggingClassifier(base = SKDecisionTreeClassifier)` |
| Spectrum | `number_of_estimators ∈ {1, 10, 50}`; `maximum_samples ∈ {all, 0.5}`; `bootstrap_features` on and off |
| Effect | bagging reduces the *variance* of a high-variance base estimator without touching its bias; the boundary of the averaged ensemble is smoother than the single tree's, and the training error rises while the test error falls — the textbook bias–variance signature of bagging |
| Output | the decision boundary per number of estimators, the training and test error, and the out-of-bag score |
| Availability | available now (`sciencekit_ensemble`) |

---

### `gradient_boosting_regression`

**Origin** — scikit-learn `examples/ensemble/plot_gradient_boosting_regression.py` · *"Gradient Boosting regression"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 10 with a non-linear ground truth |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKGradientBoostingRegressor` |
| Spectrum | `number_of_estimators ∈ {10, 50, 200}`; `learning_rate ∈ {0.01, 0.1, 1.0}`; `maximum_depth ∈ {1, 3}` |
| Effect | each new tree is fitted to the *negative gradient* of the loss — the residual of the current ensemble — so the trees specialize in what is still missing; a small learning rate needs many trees but generalizes better, a large one converges in a few trees and overshoots |
| Output | the prediction curve against the ground truth, the training loss per iteration, the feature importances, and the test MSE per configuration |
| Availability | available now (`sciencekit_ensemble`) |

---

### `early_stopping_in_gradient_boosting`

**Origin** — scikit-learn `examples/ensemble/plot_gradient_boosting_early_stopping.py` · *"Early stopping in Gradient Boosting"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 5000 × 20, deliberately noisy |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKGradientBoostingClassifier` with a validation split |
| Spectrum | the validation fraction; the number of estimators with and without early stopping |
| Effect | boosting keeps reducing the training loss long after the validation loss has started to rise; stopping at the validation optimum saves trees and avoids the overfitting that the extra trees cause |
| Output | the training and validation loss per iteration, the iteration at which the validation loss is minimal, and the score with and without stopping |
| Availability | available now (`sciencekit_ensemble`) |

---

### `regularization_of_gradient_boosting`

**Origin** — scikit-learn `examples/ensemble/plot_gradient_boosting_regularization.py` · *"Gradient Boosting regularization"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 10 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKGradientBoostingRegressor` |
| Spectrum | `l2_regularization ∈ {0.0, 0.1, 1.0, 10.0}`; `huber_delta ∈ {0.1, 1.0, 10.0}` |
| Effect | the L2 penalty shrinks each tree's leaf values, which limits how fast the ensemble can chase the residuals; the Huber delta switches the loss from quadratic to linear beyond a threshold, so the outliers stop dominating the new trees |
| Output | the training and test MSE per configuration, the number of estimators effectively used, and the coefficient magnitude of the first tree |
| Availability | available now (`sciencekit_ensemble`) |

---

### `histogram_gradient_boosting_on_wide_data`

**Origin** — scikit-learn `examples/ensemble/plot_hgbt_regression.py` · *"Features in Histogram Gradient Boosting Trees"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 100000 × 100 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKHistGradientBoostingRegressor` |
| Spectrum | the binning arm; the maximum number of bins; `maximum_depth` |
| Effect | binning the continuous features turns the split search into a histogram lookup, which is what makes the algorithm fast on wide data; the bins quantize the thresholds, so a very fine binning costs time without buying accuracy |
| Output | the fit time and the score per bin count, and the training loss curve |
| Availability | available now (`sciencekit_ensemble`) |

---

### `boosting_a_decision_tree_under_adversarial_noise`

**Origin** — scikit-learn `examples/ensemble/plot_adaboost_twoclass.py` · *"Two-class AdaBoost"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 1000 × 2 with 10 % label noise |
| Feature selection | none |
| Feature engineering | the label noise is injected |
| Preprocessing | none |
| Pipeline | `SKDecisionTreeClassifier` → an AdaBoost meta-estimator |
| Spectrum | the base estimator depth; the number of estimators; the learning rate |
| Effect | AdaBoost reweights the misclassified samples, so each successive stump concentrates on the samples the previous ones got wrong; the noise is reinforced by this mechanism — the classic failure mode of AdaBoost under label noise |
| Output | the decision boundary per number of estimators, the sample weights over the iterations, and the test error |
| Availability | planned (`sciencekit_ensemble` — the AdaBoost meta-estimator on top of the implemented trees) |

---

### `single_estimator_versus_bagging_a_bias_variance_decomposition`

**Origin** — scikit-learn `examples/ensemble/plot_bias_variance.py` · *"Single estimator versus bagging: bias-variance decomposition"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` with a sinusoidal ground truth, 100 samples, 100 repetitions of the fit |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLinearRegression` vs `SKStandardScaler → SKBaggingRegressor` |
| Spectrum | the estimator arm; the number of repetitions; the bagging size |
| Effect | the single estimator's predictions scatter widely around the truth — that scatter *is* the variance; bagging averages the scatter away and the predictions concentrate, leaving the bias unchanged |
| Output | the per-repetition predictions as a scatter band, the bias and the variance per arm, and the mean squared error decomposition |
| Availability | available now (`sciencekit_ensemble`, `sciencekit_linear_model`) |

---

### `combining_predictors_with_voting_and_stacking`

**Origin** — scikit-learn `examples/ensemble/plot_voting_decision_regions.py` and `examples/ensemble/plot_stack_predictors.py` · *"Voting" and "Stacking"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded), three classes |
| Feature selection | the two separating features |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → [SKLogisticRegression, SKSVC, SKDecisionTreeClassifier]` combined by a voting rule or by a meta-estimator |
| Spectrum | the voting rule (hard vs soft); the meta-estimator for stacking |
| Effect | hard voting takes the majority label and throws the confidence away; soft voting averages the probabilities and is usually better calibrated; stacking learns how to *weigh* the base learners from their out-of-fold predictions, which can beat both if the base learners make different mistakes |
| Output | the decision regions per combination rule, the per-class accuracy, and the out-of-fold score of the meta-estimator |
| Availability | planned (`sciencekit_ensemble` — the voting and stacking meta-estimators) |
