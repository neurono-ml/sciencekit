# 02 — Linear models

The linear-model crate is the widest implemented surface: least squares, ridge, lasso,
elastic net, logistic regression and SGD. These examples carry the book's chapters on
regularization, on sparsity and on optimization.

---

### `ordinary_least_squares_and_ridge_regression`

**Origin** — scikit-learn `examples/linear_model/plot_ols_ridge.py` · *"Ordinary Least Squares and Ridge Regression"*

| Stage | Recipe |
|---|---|
| Dataset | `load_diabetes` (embedded) — 442 × 10 numeric features, target = disease progression |
| Feature selection | none — the ten standardized predictors |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — `with_mean(true)`, `with_std(true)` |
| Pipeline | `SKStandardScaler → SKLinearRegression` and, in parallel, `SKStandardScaler → SKRidge(alpha)` |
| Spectrum | `alpha ∈ {0.0, 1.0, 10.0, 100.0}`; the `alpha = 0` arm is the OLS reference |
| Effect | as `alpha` grows the coefficients shrink toward zero; the training error rises monotonically while the test error first falls and then rises — the classic bias–variance trade visible in one table |
| Output | train and test MSE, R² and the coefficient vector per `alpha`; the coefficient path |
| Availability | available now (`sciencekit_linear_model`, `sciencekit_preprocessing`) |

---

### `ridge_coefficients_as_a_function_of_the_l2_regularization`

**Origin** — scikit-learn `examples/linear_model/plot_ridge_coeffs.py` · *"Ridge coefficients as a function of the regularization"*

| Stage | Recipe |
|---|---|
| Dataset | `load_diabetes` (embedded) |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRidge(alpha)` |
| Spectrum | a dense sweep of `alpha` over logarithmic values; and the solver arm `SKRidgeSolver ∈ {Cholesky, Svd, SparseCg}` on the dense design for `Cholesky` and `Svd` |
| Effect | the coefficient of each feature shrinks smoothly toward zero as `alpha` grows, and no coefficient ever reaches exactly zero — that is the difference from lasso; the solvers agree to numerical tolerance |
| Output | the coefficient path over `alpha`, the number of non-zero coefficients, and the solver actually used |
| Availability | available now (`sciencekit_linear_model`) |

---

### `lasso_and_elastic_net_on_sparse_signals`

**Origin** — scikit-learn `examples/linear_model/plot_lasso_and_elasticnet.py` · *"L1-based models for Sparse Signals"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 50 with only 10 informative features, the rest pure noise, plus a sparse variant of the same design |
| Feature selection | none — the estimator must discover the informative block by itself |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` (dense arm) / none (sparse arm — the CSR design is never densified) |
| Pipeline | `SKStandardScaler → SKLasso(alpha)` and `SKStandardScaler → SKElasticNet(alpha, l1_ratio)` |
| Spectrum | `alpha ∈ {0.05, 0.1, 0.5, 1.0}`; `l1_ratio ∈ {0.0, 0.5, 1.0}` for the elastic net |
| Effect | lasso drives the noisy coefficients to exactly zero and selects the informative block; the elastic net with `l1_ratio = 1` reproduces lasso, with `l1_ratio = 0` reproduces ridge, and intermediate values trade sparsity against grouping of correlated features |
| Output | the number of non-zero coefficients, the true-positive count over the ten informative features, and the test MSE |
| Availability | available now (`sciencekit_linear_model`) |

---

### `regularization_path_of_l1_logistic_regression`

**Origin** — scikit-learn `examples/linear_model/plot_logistic_path.py` · *"Regularization path of L1-Logistic Regression"*

| Stage | Recipe |
|---|---|
| Dataset | `make_multilabel_classification` — 300 × 20, three labels, correlated informative features |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression(penalty = L1)` |
| Spectrum | a dense sweep of the penalty strength; the arm `penalty = L2` as the dense reference |
| Effect | at high penalty every coefficient is exactly zero — the model predicts the prior; as the penalty falls the features enter the model one by one, in a stable order, which is the *regularization path*; the L2 arm never reaches exact zeros |
| Output | the coefficient path per class, the number of non-zero coefficients per penalty, and the training accuracy along the path |
| Availability | available now (`sciencekit_linear_model`) |

---

### `decision_boundaries_of_multinomial_and_one_vs_rest_logistic_regression`

**Origin** — scikit-learn `examples/linear_model/plot_logistic_multinomial.py` · *"Decision Boundaries of Multinomial and One-vs-Rest Logistic Regression"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 200 × 2, three classes, plotted in the original two-dimensional space |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKLogisticRegression(multiclass strategy)` |
| Spectrum | the multiclass strategy `Multinomial` vs `OneVsRest` |
| Effect | the multinomial model fits one set of weights per class that produces a consistent probability vector; the one-vs-rest model fits independent binary problems and its raw scores need normalization to be read as probabilities |
| Output | the three decision regions, the per-class probability vectors at three probe points, and the training accuracy of both arms |
| Availability | available now (`sciencekit_linear_model`) |

---

### `early_stopping_of_stochastic_gradient_descent`

**Origin** — scikit-learn `examples/linear_model/plot_sgd_early_stopping.py` · *"Early stopping of Stochastic Gradient Descent"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 5000 × 20 with a sparse informative block |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKSGDRegressor(loss)` |
| Spectrum | `loss ∈ {squared_error, epsilon_insensitive, huber}`; the learning-rate schedule and the validation-fraction arm |
| Effect | the squared error converges to the least-squares solution but keeps moving with the noise; the insensitive and Huber losses stop caring about the residuals that do not matter, which changes *where* the optimum sits and how fast it is reached |
| Output | the loss curve per iteration, the iteration at which the validation score stops improving, and the final coefficients |
| Availability | available now (`sciencekit_linear_model`) |

---

### `stochastic_gradient_descent_under_class_imbalance`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"SGD under class imbalance"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` with `weights = [0.97, 0.03]` — 3 % positives |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKSGDClassifier(class_weight)` |
| Spectrum | `class_weight ∈ {None, Balanced}`; and the threshold arm on the decision score |
| Effect | unweighted SGD minimizes the average loss, so the majority class dominates and the minority class is barely learned; balancing the classes reweights each sample and recovers the minority recall at a small cost in majority precision |
| Output | the confusion matrix, precision and recall per class, and the accuracy and balanced accuracy of both arms |
| Availability | available now (`sciencekit_linear_model`, class weighting) |

---

### `weighted_samples_in_least_squares`

**Origin** — scikit-learn `examples/linear_model/plot_sgd_weighted_samples.py` · *"SGD: Weighted samples"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 40 samples, three of them with a large residual |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | none — the example is about the weights, not the scale |
| Pipeline | `SKLinearRegression(sample_weight)` |
| Spectrum | `sample_weight ∈ {None, uniform 0.1, uniform 1.0, custom}` |
| Effect | the weighted fit is the solution of the weighted normal equations; the uniform-weight arm reproduces the unweighted fit exactly, which is the sanity check the example asserts |
| Output | the coefficients under each weighting, the weighted and unweighted residuals, and the equivalence assertion |
| Availability | available now (`sciencekit_linear_model`) |

---

### `effect_of_regularization_on_train_and_test_error`

**Origin** — scikit-learn `examples/model_selection/plot_train_error_vs_test_error.py` · *"Effect of model regularization on training and test error"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 30, ten informative features, noise added |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRidge(alpha)` |
| Spectrum | a dense sweep of `alpha` from 0 to a value where the model is constant |
| Effect | the training error rises monotonically with `alpha` while the test error has a minimum in between — the picture of the bias–variance trade-off as a single pair of curves |
| Output | the train and test error curves against `alpha`, and the `alpha` at the minimum |
| Availability | available now (`sciencekit_linear_model`, `sciencekit_metrics`) |

---

### `polynomial_and_spline_interpolation`

**Origin** — scikit-learn `examples/linear_model/plot_polynomial_interpolation.py` · *"Polynomial and Spline interpolation"*

| Stage | Recipe |
|---|---|
| Dataset | a smooth one-dimensional function sampled with noise, 200 points |
| Feature selection | the single feature |
| Feature engineering | the polynomial basis (powers of the feature) and a truncated B-spline basis |
| Preprocessing | `SKStandardScaler` on the feature before the basis expansion, to keep the powers conditioned |
| Pipeline | basis expansion → `SKLinearRegression` (polynomial) / `SKRidge` (spline, regularized) |
| Spectrum | polynomial `degree ∈ {1, 3, 7, 15}`; spline `n_knots ∈ {5, 10, 30}` with `SKRidge(alpha)` |
| Effect | a high-degree polynomial interpolates the noise and oscillates wildly between the samples, while the spline, being regularized and locally supported, follows the trend without the oscillation |
| Output | the fitted curve against the samples, the residual sum of squares, and the number of basis columns |
| Availability | available now (`sciencekit_preprocessing`, `sciencekit_linear_model`) |

---

### `quantile_regression_for_prediction_intervals`

**Origin** — scikit-learn `examples/linear_model/plot_quantile_regression.py` · *"Quantile regression"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 500 × 10 with heteroscedastic noise |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRidge` fitted at several quantile objectives |
| Spectrum | the quantile level `q ∈ {0.1, 0.5, 0.9}` |
| Effect | the median fit sits in the middle of the noise band while the 0.1 and 0.9 fits bound it; the band's width widens where the noise widens, which is exactly what a mean fit cannot express |
| Output | the three fitted lines against the samples, the empirical coverage of each interval, and the residual quantiles |
| Availability | planned (`sciencekit_linear_model` — the quantile objective on the penalized solver) |

---

### `robust_linear_estimator_under_outliers`

**Origin** — scikit-learn `examples/linear_model/plot_robust_fit.py` · *"Robust linear estimator fitting"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 100 samples with 10 % gross outliers |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKRidge` vs `SKStandardScaler → SKHuberRegressor` |
| Spectrum | the Huber `epsilon` threshold and the loss arm `SKHuberLoss` |
| Effect | least squares drags the line toward the outliers because the squared loss grows without bound; the Huber loss is linear beyond the threshold, so the outliers exert a bounded pull and the line stays where the inliers put it |
| Output | the fitted line under both estimators, the outlier list, and the number of residuals above the Huber threshold |
| Availability | planned (`sciencekit_linear_model` — the Huber estimator on top of the existing Huber loss) |

---

### `theil_sen_regression_against_ransac`

**Origin** — scikit-learn `examples/linear_model/plot_theilsen.py` · *"Theil-Sen Regression"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 50 samples with 20 % outliers |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | none |
| Pipeline | `SKLinearRegression` vs a robust estimator (Theil–Sen or RANSAC) |
| Spectrum | the estimator arm; the random seed for RANSAC's sampling |
| Effect | the median of pairwise slopes is unmoved by a minority of outliers, while least squares is dragged; the estimator's breakdown point is the point of the example |
| Output | the fitted line under each estimator, the inlier set identified, and the residual distribution |
| Availability | absent (no sciencekit path yet — the robust-estimator family is not on the roadmap) |
