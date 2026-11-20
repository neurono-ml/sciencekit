# 03 — Neighbors and support vector machines

Two families with opposite geometries: the neighbours estimators are *local* and
instance-based, the support vector machines are *global* and margin-based. Their
examples answer the same question from opposite ends — how much does the geometry of
the problem decide the model?

---

### `recognizing_handwritten_digits_with_a_neighbor_classifier`

**Origin** — scikit-learn `examples/classification/plot_digits_classification.py` · *"Recognizing hand-written digits"*

| Stage | Recipe |
|---|---|
| Dataset | `load_digits` (embedded) — 1797 × 64 pixel intensities, ten classes |
| Feature selection | none — the full 64-pixel image |
| Feature engineering | none — the pixels *are* the features |
| Preprocessing | `SKStandardScaler` — the intensities are scaled per pixel, which is what makes euclidean distance meaningful |
| Pipeline | `SKStandardScaler → SKKNeighborsClassifier` |
| Spectrum | `nearest_neighbors_count ∈ {1, 3, 5, 10, 20}`; the search-strategy arm `BruteForce` vs `KDTree` vs `BallTree` |
| Effect | with a single neighbour the prediction is the label of the closest training image and the error rate is highest; more neighbours smooth the decision boundary and the error falls, then flattens; the tree indexes and the brute-force searcher agree exactly, which is the parity assertion |
| Output | the confusion matrix, the per-class accuracy, and the agreement between the search strategies |
| Availability | available now (`sciencekit_neighbors`, `sciencekit_preprocessing`) |

---

### `k_neighbor_regression_on_a_noisy_function`

**Origin** — scikit-learn `examples/neighbors/plot_neighbors_regression.py` · *"Comparing ordinary and robust KNeighbors Regressor"*

| Stage | Recipe |
|---|---|
| Dataset | a one-dimensional function sampled with noise, 200 points |
| Feature selection | the single feature |
| Feature engineering | none |
| Preprocessing | none |
| Pipeline | `SKKNeighborsRegressor` |
| Spectrum | `nearest_neighbors_count ∈ {1, 5, 15}`; the weighting arm (uniform vs distance) |
| Effect | a single neighbour reproduces the training points exactly and the curve is jagged; more neighbours average the noise away and the curve follows the trend; distance weighting pulls the prediction towards the closer samples |
| Output | the fitted curve against the samples, the mean absolute error, and the number of neighbours used |
| Availability | available now (`sciencekit_neighbors`) |

---

### `choosing_the_metric_for_the_neighbor_search`

**Origin** — scikit-learn `examples/neighbors/plot_neighbors_metric.py` · *"Comparing metrics in K-Nearest Neighbors"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 300 × 4, two informative features, two noise features, scaled to incomparable magnitudes |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — without it, the euclidean metric is dominated by the large-scale feature |
| Pipeline | `SKStandardScaler → SKKNeighborsClassifier(metric)` |
| Spectrum | `metric ∈ {euclidean, manhattan, chebyshev, minkowski}` |
| Effect | on data whose features live on different scales, the euclidean distance is effectively a one-feature distance; after scaling, the metrics agree more closely, and their remaining differences expose the geometry they assume (L1 versus L∞ versus L2) |
| Output | the decision boundary per metric, the test accuracy per metric, and the pairwise distance matrix of a few probe points |
| Availability | available now (`sciencekit_neighbors`, `sciencekit_math` pairwise) |

---

### `sparse_input_never_densified_in_the_neighbor_search`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Representation parity in the neighbour estimators"*

| Stage | Recipe |
|---|---|
| Dataset | `make_sparse_spd_matrix` — a sparse symmetric positive-definite design with 5 % density, plus its dense twin |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | none — the sparse design is scaled in place, never densified |
| Pipeline | `SKKNeighborsClassifier` on the sparse design vs on the dense twin |
| Spectrum | the representation arm (dense / sparse); the search-strategy arm |
| Effect | the sparse and dense fits agree within tolerance, and the sparse path never materializes a dense matrix — the representation-parity contract made observable |
| Output | the sparse storage size against the dense one, the agreement of the predictions, and the peak memory of the fit |
| Availability | available now (`sciencekit_neighbors`, `sciencekit_common` data views) |

---

### `support_vector_classifier_with_a_radial_kernel`

**Origin** — scikit-learn `examples/svm/plot_svm_iris.py` · *"SVM: Iris 3-class classification"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) — 150 × 4, three classes |
| Feature selection | the two features that separate the classes: sepal length and petal length |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — the kernel's distances are meaningless on unstandardized features |
| Pipeline | `SKStandardScaler → SKSVC(kernel)` |
| Spectrum | `kernel ∈ {linear, polynomial, radial_basis_function, sigmoid}`; `C ∈ {0.1, 1.0, 10.0, 100.0}`; `gamma` per kernel |
| Effect | the linear kernel cannot separate the overlapping classes and mislabels the region where two classes meet; the radial kernel carves out a bounded island around each class; a large `C` makes the margin narrow and the model brittle around the outliers, while a small `C` tolerates them at the price of a wider margin |
| Output | the decision regions, the support vectors per class, the margin width, and the test accuracy per configuration |
| Availability | available now (`sciencekit_svm`) |

---

### `support_vector_regressor_against_the_kernel_ridge_regressor`

**Origin** — scikit-learn `examples/miscellaneous/plot_kernel_ridge_regression.py` · *"Comparison of kernel ridge regression and SVR"*

| Stage | Recipe |
|---|---|
| Dataset | a one-dimensional function sampled with noise, 200 points |
| Feature selection | the single feature |
| Feature engineering | none |
| Preprocessing | none |
| Pipeline | `SKSVR(kernel)` vs a kernel ridge regressor |
| Spectrum | `C ∈ {0.1, 1.0, 100.0}` for the SVR; `epsilon ∈ {0.01, 0.1, 0.5}` |
| Effect | the epsilon-insensitive loss ignores errors smaller than `epsilon`, so the fit is flat where the data is within the tube and bends only outside it; `C` then decides how much the tube may be violated |
| Output | the fitted curve and the epsilon tube against the samples, the number of support vectors, and the mean absolute error |
| Availability | available now (`sciencekit_svm`); the kernel ridge arm is planned (`sciencekit_linear_model`) |

---

### `linear_support_vector_classifier_under_sparse_input`

**Origin** — scikit-learn `examples/svm/plot_svm_linear.py` · *"SVM: Linear SVM"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) with two classes, and a sparse variant of the same design |
| Feature selection | the two separating features |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` on the dense arm; the sparse arm is scaled in place |
| Pipeline | `SKStandardScaler → SKLinearSVC` |
| Spectrum | `C ∈ {0.01, 0.1, 1.0, 10.0}`; the representation arm (dense / sparse) |
| Effect | the linear model draws a straight boundary, which is exactly enough for this dataset and far cheaper than the kernel machine; as `C` grows the boundary is pulled to classify the hard points correctly, at the cost of a narrower margin and a larger risk of overfitting |
| Output | the decision line, the margin width, the number of support vectors, and the agreement between the dense and sparse fits |
| Availability | available now (`sciencekit_svm`) |

---

### `one_class_support_vector_machine_as_an_outlier_detector`

**Origin** — scikit-learn `examples/svm/plot_one_class_svm.py` · *"One-Class SVM versus One-Class SVM using Stochastic Gradient Descent"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` — one dense cluster plus 5 % uniform outliers |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKSVC(one-class)` |
| Spectrum | `nu ∈ {0.05, 0.1, 0.5}`; `kernel ∈ {linear, radial_basis_function}` |
| Effect | `nu` is the upper bound on the fraction of outliers and also the lower bound on the fraction of support vectors; the linear kernel can only enclose the cluster in a half-space, while the radial kernel can wrap it in a bounded region — the shape of the boundary is the kernel's geometry |
| Output | the decision surface, the identified outliers, the number of support vectors, and the outlier fraction against the requested `nu` |
| Availability | planned (`sciencekit_outlier` — the one-class SVM, tracked with the LOF work) |

---

### `support_vector_machines_under_a_nonlinear_boundary`

**Origin** — scikit-learn `examples/svm/plot_svm_margin.py` · *"SVM: Margin effect"*

| Stage | Recipe |
|---|---|
| Dataset | `make_moons` — two interleaved half-moons, not linearly separable |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKSVC(kernel, C)` |
| Spectrum | `kernel ∈ {linear, radial_basis_function}`; `C ∈ {0.1, 1.0, 10.0}`; `gamma ∈ {0.1, 1.0, 10.0}` |
| Effect | the linear kernel must cut one moon in half to fit the data; the radial kernel bends around it; `gamma` controls how far the influence of one training point reaches — too small and the boundary is too smooth, too large and it becomes a set of islands around each point |
| Output | the decision regions per configuration, the test accuracy, and the number of support vectors |
| Availability | available now (`sciencekit_svm`) |

---

### `probability_calibration_of_a_support_vector_classifier`

**Origin** — scikit-learn `examples/calibration/plot_calibration_curve.py` · *"Probability Calibration curves"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 2000 × 20 with a deliberately miscalibrated score distribution |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKSVC` with a probability model over the decision function |
| Spectrum | the calibration method (sigmoid vs isotonic); `C` as the base arm |
| Effect | the raw decision function is not a probability: it is unbounded and only its sign is meaningful; the Platt scaling fits a sigmoid to the scores and makes them readable as probabilities, at the price of a monotone distortion |
| Output | the reliability curve against the diagonal, the calibration error, and the log loss before and after |
| Availability | planned (`sciencekit_metrics` — the calibration work is planned with the calibration and multilabel metrics change) |
