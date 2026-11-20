# 10 — Outliers, covariance and the long tail

The families sciencekit does not cover yet. They are here so the catalog's omissions
are *decisions* with a recorded reason, not gaps nobody looked at. Each record states
what the example would teach and what sciencekit would have to grow first.

---

### `isolation_forest_as_an_outlier_detector`

**Origin** — scikit-learn `examples/ensemble/plot_isolation_forest.py` · *"IsolationForest example"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` — one dense cluster plus 5 % uniform outliers |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKIsolationForest` |
| Spectrum | `contamination ∈ {0.05, 0.1, 0.2}`; `maximum_samples ∈ {64, 128, 256}`; `number_of_estimators` |
| Effect | the isolation forest does not model the distribution — it measures how *isolated* a point is in a random partition of the feature space, so a point that needs few splits to be separated is an outlier; the `contamination` parameter sets how many points are declared anomalous, and the score is only comparable *within* one fitted model |
| Output | the anomaly scores, the identified outliers, the precision and recall against the injected outliers, and the score distribution |
| Availability | available now (`sciencekit_outlier`) |

---

### `comparing_anomaly_detection_estimators`

**Origin** — scikit-learn `examples/miscellaneous/plot_anomaly_comparison.py` · *"Comparing anomaly detection algorithms for outlier detection on toy datasets"*

| Stage | Recipe |
|---|---|
| Dataset | four shapes: a Gaussian blob, a uniform box, a clustered set and a sinusoidal manifold, each with injected anomalies |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKIsolationForest` vs the local-outlier and one-class estimators |
| Spectrum | the estimator arm; the contamination arm |
| Effect | the estimators disagree because they assume different things: isolation forest assumes sparsity and local isolation, the local-outlier factor assumes local density, the one-class SVM assumes a bounded region around the bulk — on the sinusoidal manifold the first two fail because every point is equally isolated, and only a method that models the *shape* of the data succeeds |
| Output | the detected anomalies per estimator per shape, the precision and recall per combination, and the estimator's assumption stated for each |
| Availability | planned (`sciencekit_outlier` — the local-outlier factor and the one-class SVM are tracked in the outlier LOF and one-class SVM change) |

---

### `outlier_detection_on_a_real_dataset`

**Origin** — scikit-learn `examples/applications/plot_outlier_detection_wine.py` · *"Outlier detection on a real data set"*

| Stage | Recipe |
|---|---|
| Dataset | `load_wine` (embedded) — 178 × 13, three cultivars |
| Feature selection | the two features with the largest variance, after standardization |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKIsolationForest` |
| Spectrum | the feature subset; the contamination arm |
| Effect | without standardizing, the outlier score is computed mostly from the two wide features and the narrow ones are ignored; after standardization every feature contributes, and the flagged samples are the ones that are extreme in *several* dimensions at once |
| Output | the flagged samples with their feature values, the score distribution per cultivar, and the flagged samples per dimension |
| Availability | available now (`sciencekit_outlier`, `sciencekit_preprocessing`) |

---

### `empirical_against_shrunk_covariance`

**Origin** — scikit-learn `examples/covariance/plot_covariance_estimation.py` · *"Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 100 samples × 50 features, so the sample covariance is singular |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → covariance estimator` |
| Spectrum | the estimator arm (empirical, Ledoit–Wolf, OAS) |
| Effect | with fewer samples than features the empirical covariance is singular and its inverse does not exist; the shrunk estimators pull the estimate towards a scaled identity and keep it invertible, at the price of a bias towards the diagonal — the shrinkage intensity is the knob that trades variance for bias |
| Output | the estimated covariance, its condition number, the shrinkage coefficient, and the reconstruction error against the empirical estimate |
| Availability | absent (no sciencekit path yet — the covariance family is not on the roadmap) |

---

### `mahalanobis_distances_under_a_robust_covariance`

**Origin** — scikit-learn `examples/covariance/plot_mahalanobis_distances.py` · *"Robust covariance estimation and Mahalanobis distances relevance"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` with 5 % gross outliers |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → covariance estimator → Mahalanobis distance` |
| Spectrum | the covariance arm (empirical against robust) |
| Effect | the empirical covariance is itself dragged by the outliers, so the outliers end up *inside* the ellipse they should be outside; the robust estimate keeps the ellipse tight and the anomalies outside — the estimator of the covariance decides the detector |
| Output | the Mahalanobis distances, the ellipse in two dimensions, and the fraction of outliers beyond a threshold under each covariance |
| Availability | absent (no sciencekit path yet) |

---

### `gaussian_mixture_model_of_a_density`

**Origin** — scikit-learn `examples/mixture/plot_gmm_pdf.py` · *"Density Estimation for a Gaussian mixture"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` with two clusters of different variances |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → Gaussian mixture model` |
| Spectrum | the number of components ∈ {1, 2, 3}; the covariance arm (full, tied, diagonal, spherical) |
| Effect | the mixture model estimates a *density*, so it can say how likely any point is and which component generated it; the covariance type is the model's capacity — full covariance can model an elongated cluster, diagonal cannot, and the difference shows up exactly on anisotropic data |
| Output | the density over the grid, the component weights, the per-component means and covariances, and the likelihood per component count |
| Availability | absent (no sciencekit path yet — the mixture family is not on the roadmap) |

---

### `gaussian_process_regression_of_a_noisy_function`

**Origin** — scikit-learn `examples/gaussian_process/plot_gpr_noisy_targets.py` · *"Gaussian Processes regression: basic introductory example"*

| Stage | Recipe |
|---|---|
| Dataset | a smooth function sampled with noise, 100 points |
| Feature selection | the single feature |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` on the feature |
| Pipeline | `SKStandardScaler → Gaussian process regressor(kernel)` |
| Spectrum | the kernel arm (RBF, Matérn, white); the noise level |
| Effect | the kernel *is* the prior over the function: the RBF prior says the function is smooth over a length scale, the white component says it is noisy; the posterior is the prior conditioned on the observations, so the fit interpolates the data and the uncertainty grows where the data is sparse |
| Output | the posterior mean and the confidence band, the log marginal likelihood per kernel, and the length scale and noise learned |
| Availability | absent (no sciencekit path yet) |

---

### `neural_network_classifier_on_a_synthetic_boundary`

**Origin** — scikit-learn `examples/neural_networks/plot_mlp_classifier.py` · *"Plot a decision function for a multilayer perceptron"*

| Stage | Recipe |
|---|---|
| Dataset | `make_moons` — a non-linear boundary |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → multi-layer perceptron` |
| Spectrum | the hidden-layer-size arm (none, 10, 100); the activation arm; the regularization arm |
| Effect | a perceptron with no hidden layer is a linear model and cannot fit the moons; one hidden layer with enough units can, but its width and its weight decay decide whether it fits the trend or the noise |
| Output | the decision function, the training loss curve, the weight norms per layer, and the score per architecture |
| Availability | absent (no sciencekit path yet — the neural-network family is not on the roadmap) |

---

### `text_documents_as_a_sparse_design`

**Origin** — scikit-learn `examples/text/plot_document_classification_20newsgroups.py` · *"Text classification"*

| Stage | Recipe |
|---|---|
| Dataset | a small embedded corpus of documents with two labels — the full 20-newsgroups corpus is not embedded |
| Feature selection | the vocabulary restricted to the most frequent terms, and the documents restricted to the most frequent label |
| Feature engineering | the term-frequency weighting with the sublinear scaling and the stop-word removal |
| Preprocessing | the sparse normalization; the design is never densified |
| Pipeline | sparse weighting → `SKLinearSVC` or `SKLogisticRegression` |
| Spectrum | the vocabulary size ∈ {1000, 5000, 20000}; the classifier arm; the weighting arm |
| Effect | the vocabulary size is the model's capacity in disguise: a small vocabulary cannot express the distinction between the classes that lives in the rare terms, and a large one overfits the corpus's peculiarities; the linear model works because text is high-dimensional and sparse |
| Output | the vocabulary size, the per-class F1, the confusion matrix, and the memory footprint against the dense path |
| Availability | planned (`sciencekit_examples` — the vectorizers are example-local; the estimator is available now) |

---

### `naive_bayes_for_text_categorical_data`

**Origin** — scikit-learn `examples/text/plot_document_classification_20newsgroups.py` · *"Naive Bayes" arm |

| Stage | Recipe |
|---|---|
| Dataset | the same embedded corpus |
| Feature selection | the vocabulary |
| Feature engineering | the term counts |
| Preprocessing | the count normalization |
| Pipeline | counts → multinomial naive Bayes |
| Spectrum | the smoothing arm (Laplace against none) |
| Effect | the naive Bayes model assumes the features are independent given the class, which is false for text — and yet it works, because the assumption is only used to *factorize* a posterior whose ranking it preserves often enough; the smoothing is what keeps an unseen term from zeroing the whole posterior |
| Output | the per-class F1 against the linear model, the log-probability of a document under each class, and the effect of removing the smoothing |
| Availability | absent (no sciencekit path yet — the naive Bayes family is not on the roadmap) |

---

### `semi_supervised_learning_with_labeled_and_unlabeled_samples`

**Origin** — scikit-learn `examples/semi_supervised/plot_label_spreading.py` · *"Label Spreading"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` with 10 % labeled and 90 % unlabeled samples |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → label spreading` |
| Spectrum | the number of labeled samples ∈ {10, 50, 200}; the kernel arm |
| Effect | the method spreads the few labels along the graph of the data, so the *geometry* carries the information to the unlabeled points; with too few labels the propagation follows the wrong structure and the error is worse than the supervised baseline trained on the same labels |
| Output | the propagated labels, the accuracy against the true labels, and the accuracy of the supervised baseline on the same labeled subset |
| Availability | absent (no sciencekit path yet) |

---

### `multiclass_strategy_for_a_binary_only_estimator`

**Origin** — scikit-learn `examples/multiclass/plot_multiclass.py` · *"Multiclass classification"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) — three classes |
| Feature selection | the two separating features |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → binary estimator → multiclass strategy` |
| Spectrum | the strategy arm (one-vs-rest against one-vs-one); the estimator arm |
| Effect | one-vs-rest trains three binary problems and predicts the highest score, which can be inconsistent when two classes both claim a point; one-vs-one trains one problem per pair and votes, which is more balanced but costs quadratically in the number of classes |
| Output | the decision regions, the per-class accuracy, the number of fitted models, and the training time |
| Availability | planned (`sciencekit_common` — the multiclass strategy on top of the existing predictor contracts) |

---

### `multioutput_regression_with_shared_structure`

**Origin** — scikit-learn `examples/multioutput/plot_multioutput_face_completion.py` · *"Face completion with a multi-output estimator"*

| Stage | Recipe |
|---|---|
| Dataset | a synthetic image-like matrix — 1000 samples × 400 pixels, arranged so the pixels form a low-rank structure |
| Feature selection | none |
| Feature engineering | the target is the same matrix shifted, so the task is to predict a missing block |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → multi-output regressor` |
| Spectrum | the estimator arm (a single multi-output tree against one regressor per target) |
| Effect | the multi-output tree splits *once* per node and predicts all targets, so the targets share the split structure and the model is far cheaper; the per-target approach ignores the correlation between the targets and pays for it in accuracy |
| Output | the reconstruction error per target, the total training time of both arms, and the correlation structure exploited |
| Availability | absent (no sciencekit path yet) |

---

### `cross_decomposition_of_two_views`

**Origin** — scikit-learn `examples/cross_decomposition/plot_compare_cross_decomposition.py` · *"Compare cross decomposition methods"*

| Stage | Recipe |
|---|---|
| Dataset | two views of the same samples — a measurement matrix and its noisy image |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` on each view |
| Pipeline | `SKStandardScaler(view A) ∥ SKStandardScaler(view B) → cross decomposition` |
| Spectrum | the method arm (principal component regression against partial least squares); the number of components |
| Effect | principal component regression finds the directions of the *first* view and then regresses the second on them; partial least squares finds the directions that best *predict* the second view, which is a different objective and a different answer |
| Output | the components of both views, the cross-correlation between the components, and the reconstruction error of each view |
| Availability | absent (no sciencekit path yet) |

---

### `biclustering_of_a_block_structure`

**Origin** — scikit-learn `examples/bicluster/plot_spectral_coclustering.py` · *"A demo of the Spectral Co-Clustering algorithm"*

| Stage | Recipe |
|---|---|
| Dataset | a checkerboard matrix — rows and columns grouped, with a block structure |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` on the matrix |
| Pipeline | `SKStandardScaler → spectral co-clustering` |
| Spectrum | the number of row and column clusters |
| Effect | co-clustering searches for the *pairs* of row and column groups that are mutually informative, which is a different problem from clustering the rows alone; the answer is a rectangle of the matrix, not a list of rows |
| Output | the row and column labels, the identified blocks, and the agreement with the planted structure |
| Availability | absent (no sciencekit path yet) |

---

### `kernel_approximation_for_scalable_learning`

**Origin** — scikit-learn `examples/kernel_approximation/plot_scalable_poly_kernels.py` · *"Scalable learning with polynomial kernel approximation"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 20000 × 50 |
| Feature selection | none |
| Feature engineering | the random feature map approximating the polynomial kernel |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → random features → SKLogisticRegression` |
| Spectrum | the component count ∈ {100, 500, 2000}; the degree |
| Effect | the exact kernel machine scales quadratically with the sample count because it must keep every training vector; the explicit map scales linearly and loses only the approximation error — the trade is accuracy for linear time |
| Output | the approximation error against the exact kernel, the training time of both arms, and the test score |
| Availability | planned (`sciencekit_decomposition` — the kernel-approximation family) |

---

### `calibration_of_a_classifier_under_distribution_shift`

**Origin** — scikit-learn `examples/calibration/plot_compare_calibration.py` · *"Comparison of Calibration of Classifiers"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` trained on one regime and evaluated on a shifted one |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → classifier → calibration` |
| Spectrum | the classifier arm; the calibration arm (sigmoid against isotonic) |
| Effect | a model trained on one regime and deployed on another keeps its ranking but loses its calibration, because the base rate changed; the sigmoid correction is stable and assumes a logistic relation, the isotonic correction is flexible and can overfit a small calibration set |
| Output | the reliability curve before and after, the expected calibration error, and the log loss |
| Availability | planned (`sciencekit_metrics` — the calibration metrics are planned with the calibration and multilabel metrics change) |

---

### `out_of_core_classification_of_a_stream`

**Origin** — scikit-learn `examples/applications/plot_out_of_core_classification.py` · *"Out-of-core classification of text documents"*

| Stage | Recipe |
|---|---|
| Dataset | a synthetic document stream — 100000 documents delivered in blocks |
| Feature selection | the vocabulary fitted on the first block only |
| Feature engineering | the term-frequency weighting applied per block |
| Preprocessing | `partial_fit` of the scaler and the classifier over the stream |
| Pipeline | stream → `partial_fit(SKStandardScaler)` → `partial_fit(SKSGDClassifier)` |
| Spectrum | the block size; the epoch count; the batch arm against a batch fit |
| Effect | the incremental fit sees each document once and never stores it, so the memory is bounded by the block; the price is that it cannot revisit an earlier block, and the order of the stream therefore matters to the result |
| Output | the memory high-water mark, the accuracy on the final block, and the score of the batch arm |
| Availability | available now (`sciencekit_common` streaming contracts, `sciencekit_linear_model`) |

---

### `outlier_detection_in_a_feature_space_with_mixed_scales`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Scale and the anomaly score"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` in two dimensions, one multiplied by 1000, with injected anomalies |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` (or not) |
| Pipeline | `SKIsolationForest` on the raw design against the standardized one |
| Spectrum | the scaling arm; the contamination arm |
| Effect | the isolation forest splits on random axes chosen uniformly from the feature range, so a feature with a huge range is split almost exclusively and a small one is effectively ignored; without standardization the anomaly score is a function of one dimension only |
| Output | the anomaly scores under both arms, the overlap of the detected sets, and the precision and recall against the injected anomalies |
| Availability | available now (`sciencekit_outlier`, `sciencekit_preprocessing`) |
