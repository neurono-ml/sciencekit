# Availability matrix

Every FDR of the corpus, its origin and its state. The state is the same
vocabulary the catalog uses, so the matrix and the catalog cannot disagree:
`available now` means the sciencekit capability exists today; `planned` names
the crate that must grow first; `absent` means the family is not on the
roadmap and the example is a deliberate omission with a recorded reason.

| Family | Examples | available now | planned | absent |
|---|---|---|---|---|
| [Preprocessing and imputation](01-preprocessing-and-imputation.md) | 11 | 11 | 0 | 0 |
| [Linear models](02-linear-models.md) | 13 | 10 | 2 | 1 |
| [Neighbors and support vector machines](03-neighbors-and-svm.md) | 10 | 8 | 2 | 0 |
| [Trees and ensembles](04-trees-and-ensembles.md) | 14 | 11 | 3 | 0 |
| [Clustering](05-clustering.md) | 12 | 8 | 4 | 0 |
| [Decomposition](06-decomposition.md) | 13 | 6 | 6 | 1 |
| [Model selection and pipelines](07-model-selection-and-pipelines.md) | 11 | 7 | 4 | 0 |
| [Metrics and inspection](08-metrics-and-inspection.md) | 11 | 5 | 6 | 0 |
| [Feature selection and engineering](09-feature-selection-and-engineering.md) | 11 | 1 | 10 | 0 |
| [Outliers, covariance and the long tail](10-outliers-covariance-and-long-tail.md) | 19 | 4 | 5 | 10 |
| **Total** | **125** | **71** | **42** | **12** |

## The state of every example

| Example | Origin | State |
|---|---|---|
| `standard_scaler_on_a_mixed_scale_dataset` | scikit-learn examples/preprocessing/plot_scaling.py · "Standardize a feature by removing the mean and scaling to unit variance" | available now (`sciencekit_preprocessing`, `sciencekit_linear_model`) |
| `minmax_scaler_to_a_bounded_range` | scikit-learn examples/preprocessing/plot_scaling.py · "MinMaxScaler" | available now (`sciencekit_preprocessing`) |
| `robust_scaler_against_outliers` | scikit-learn examples/preprocessing/plot_scaling.py · "RobustScaler" | available now (`sciencekit_preprocessing`) |
| `one_hot_encoding_of_categorical_features` | scikit-learn examples/preprocessing/plot_one_hot_encoder.py · "One-hot encoding of categorical features" | available now (`sciencekit_preprocessing`) |
| `polynomial_feature_expansion_of_interactions` | scikit-learn examples/preprocessing/plot_polynomial_features.py · "Polynomial expansion" | available now (`sciencekit_preprocessing`) |
| `imputing_missing_values_before_building_an_estimator` | scikit-learn examples/impute/plot_missing_values.py · "Imputing missing values before building an estimator" | available now (`sciencekit_impute`, `sciencekit_linear_model`) |
| `missing_indicator_as_an_explicit_signal` | scikit-learn examples/impute/plot_missing_values.py · "Missing-indicator" arm of the imputation example" | available now (`sciencekit_impute`) |
| `knn_imputation_of_missing_feature_values` | scikit-learn examples/impute/plot_missing_values.py · "Nearest-neighbor imputation" | available now (`sciencekit_impute`, `sciencekit_neighbors`) |
| `scaling_inside_a_cross_validated_pipeline` | scikit-learn examples/preprocessing/plot_scaling.py · "Scaling inside a cross-validated pipeline" | available now (`sciencekit_model_selection`, `sciencekit_preprocessing`) |
| `streaming_scaler_under_a_shifting_distribution` | no scikit-learn origin — a sciencekit-native extension · "Incremental standardization under a drifting distribution" | available now (`sciencekit_preprocessing`, `sciencekit_common` streaming contracts) |
| `preprocessing_chain_cheat_sheet` | no scikit-learn origin — a sciencekit-native reference page · "The preprocessing chain, end to end" | available now (`sciencekit_pipeline`, `sciencekit_impute`, `sciencekit_preprocessing`) |
| `ordinary_least_squares_and_ridge_regression` | scikit-learn examples/linear_model/plot_ols_ridge.py · "Ordinary Least Squares and Ridge Regression" | available now (`sciencekit_linear_model`, `sciencekit_preprocessing`) |
| `ridge_coefficients_as_a_function_of_the_l2_regularization` | scikit-learn examples/linear_model/plot_ridge_coeffs.py · "Ridge coefficients as a function of the regularization" | available now (`sciencekit_linear_model`) |
| `lasso_and_elastic_net_on_sparse_signals` | scikit-learn examples/linear_model/plot_lasso_and_elasticnet.py · "L1-based models for Sparse Signals" | available now (`sciencekit_linear_model`) |
| `regularization_path_of_l1_logistic_regression` | scikit-learn examples/linear_model/plot_logistic_path.py · "Regularization path of L1-Logistic Regression" | available now (`sciencekit_linear_model`) |
| `decision_boundaries_of_multinomial_and_one_vs_rest_logistic_regression` | scikit-learn examples/linear_model/plot_logistic_multinomial.py · "Decision Boundaries of Multinomial and One-vs-Rest Logistic Regression" | available now (`sciencekit_linear_model`) |
| `early_stopping_of_stochastic_gradient_descent` | scikit-learn examples/linear_model/plot_sgd_early_stopping.py · "Early stopping of Stochastic Gradient Descent" | available now (`sciencekit_linear_model`) |
| `stochastic_gradient_descent_under_class_imbalance` | no scikit-learn origin — a sciencekit-native extension · "SGD under class imbalance" | available now (`sciencekit_linear_model`, class weighting) |
| `weighted_samples_in_least_squares` | scikit-learn examples/linear_model/plot_sgd_weighted_samples.py · "SGD: Weighted samples" | available now (`sciencekit_linear_model`) |
| `effect_of_regularization_on_train_and_test_error` | scikit-learn examples/model_selection/plot_train_error_vs_test_error.py · "Effect of model regularization on training and test error" | available now (`sciencekit_linear_model`, `sciencekit_metrics`) |
| `polynomial_and_spline_interpolation` | scikit-learn examples/linear_model/plot_polynomial_interpolation.py · "Polynomial and Spline interpolation" | available now (`sciencekit_preprocessing`, `sciencekit_linear_model`) |
| `quantile_regression_for_prediction_intervals` | scikit-learn examples/linear_model/plot_quantile_regression.py · "Quantile regression" | planned (`sciencekit_linear_model` — the quantile objective on the penalized solver) |
| `robust_linear_estimator_under_outliers` | scikit-learn examples/linear_model/plot_robust_fit.py · "Robust linear estimator fitting" | planned (`sciencekit_linear_model` — the Huber estimator on top of the existing Huber loss) |
| `theil_sen_regression_against_ransac` | scikit-learn examples/linear_model/plot_theilsen.py · "Theil-Sen Regression" | absent (no sciencekit path yet — the robust-estimator family is not on the roadmap) |
| `recognizing_handwritten_digits_with_a_neighbor_classifier` | scikit-learn examples/classification/plot_digits_classification.py · "Recognizing hand-written digits" | available now (`sciencekit_neighbors`, `sciencekit_preprocessing`) |
| `k_neighbor_regression_on_a_noisy_function` | scikit-learn examples/neighbors/plot_neighbors_regression.py · "Comparing ordinary and robust KNeighbors Regressor" | available now (`sciencekit_neighbors`) |
| `choosing_the_metric_for_the_neighbor_search` | scikit-learn examples/neighbors/plot_neighbors_metric.py · "Comparing metrics in K-Nearest Neighbors" | available now (`sciencekit_neighbors`, `sciencekit_math` pairwise) |
| `sparse_input_never_densified_in_the_neighbor_search` | no scikit-learn origin — a sciencekit-native extension · "Representation parity in the neighbour estimators" | available now (`sciencekit_neighbors`, `sciencekit_common` data views) |
| `support_vector_classifier_with_a_radial_kernel` | scikit-learn examples/svm/plot_svm_iris.py · "SVM: Iris 3-class classification" | available now (`sciencekit_svm`) |
| `support_vector_regressor_against_the_kernel_ridge_regressor` | scikit-learn examples/miscellaneous/plot_kernel_ridge_regression.py · "Comparison of kernel ridge regression and SVR" | available now (`sciencekit_svm`); the kernel ridge arm is planned (`sciencekit_linear_model`) |
| `linear_support_vector_classifier_under_sparse_input` | scikit-learn examples/svm/plot_svm_linear.py · "SVM: Linear SVM" | available now (`sciencekit_svm`) |
| `one_class_support_vector_machine_as_an_outlier_detector` | scikit-learn examples/svm/plot_one_class_svm.py · "One-Class SVM versus One-Class SVM using Stochastic Gradient Descent" | planned (`sciencekit_outlier` — the one-class SVM, tracked with the LOF work) |
| `support_vector_machines_under_a_nonlinear_boundary` | scikit-learn examples/svm/plot_svm_margin.py · "SVM: Margin effect" | available now (`sciencekit_svm`) |
| `probability_calibration_of_a_support_vector_classifier` | scikit-learn examples/calibration/plot_calibration_curve.py · "Probability Calibration curves" | planned (`sciencekit_metrics` — the calibration work is planned with the calibration and multilabel metrics change) |
| `decision_surface_of_a_decision_tree` | scikit-learn examples/tree/plot_iris_dtc.py · "Decision surface of decision trees trained on the iris dataset" | available now (`sciencekit_tree`) |
| `understanding_the_structure_of_a_decision_tree` | scikit-learn examples/tree/plot_unveil_tree_structure.py · "Understanding the decision tree structure" | available now (`sciencekit_tree`) |
| `post_pruning_a_decision_tree_by_cost_complexity` | scikit-learn examples/tree/plot_cost_complexity_pruning.py · "Post pruning decision trees with cost complexity pruning" | planned (`sciencekit_tree` — the cost-complexity pruning on top of the implemented tree) |
| `random_forest_on_the_iris_dataset` | scikit-learn examples/ensemble/plot_forest_iris.py · "Decision surfaces of ensembles of trees on the iris dataset" | available now (`sciencekit_ensemble`) |
| `out_of_bag_errors_for_random_forests` | scikit-learn examples/ensemble/plot_ensemble_oob.py · "OOB Errors for Random Forests" | available now (`sciencekit_ensemble`) |
| `feature_importances_of_a_forest_of_trees` | scikit-learn examples/ensemble/plot_forest_importances.py · "Feature importances with a forest of trees" | available now (`sciencekit_ensemble` for the impurity importances; the permutation arm is planned in `sciencekit_metrics`) |
| `bagging_a_base_estimator_on_bootstrap_samples` | scikit-learn examples/ensemble/plot_bagging_classifier.py · "Bagging classifier" | available now (`sciencekit_ensemble`) |
| `gradient_boosting_regression` | scikit-learn examples/ensemble/plot_gradient_boosting_regression.py · "Gradient Boosting regression" | available now (`sciencekit_ensemble`) |
| `early_stopping_in_gradient_boosting` | scikit-learn examples/ensemble/plot_gradient_boosting_early_stopping.py · "Early stopping in Gradient Boosting" | available now (`sciencekit_ensemble`) |
| `regularization_of_gradient_boosting` | scikit-learn examples/ensemble/plot_gradient_boosting_regularization.py · "Gradient Boosting regularization" | available now (`sciencekit_ensemble`) |
| `histogram_gradient_boosting_on_wide_data` | scikit-learn examples/ensemble/plot_hgbt_regression.py · "Features in Histogram Gradient Boosting Trees" | available now (`sciencekit_ensemble`) |
| `boosting_a_decision_tree_under_adversarial_noise` | scikit-learn examples/ensemble/plot_adaboost_twoclass.py · "Two-class AdaBoost" | planned (`sciencekit_ensemble` — the AdaBoost meta-estimator on top of the implemented trees) |
| `single_estimator_versus_bagging_a_bias_variance_decomposition` | scikit-learn examples/ensemble/plot_bias_variance.py · "Single estimator versus bagging: bias-variance decomposition" | available now (`sciencekit_ensemble`, `sciencekit_linear_model`) |
| `combining_predictors_with_voting_and_stacking` | scikit-learn examples/ensemble/plot_voting_decision_regions.py and examples/ensemble/plot_stack_predictors.py · "Voting" and "Stacking" | planned (`sciencekit_ensemble` — the voting and stacking meta-estimators) |
| `k_means_clustering_on_a_blobs_dataset` | scikit-learn examples/cluster/plot_kmeans_blobs.py · "Demonstration of k-means assumptions" | available now (`sciencekit_cluster`, `sciencekit_preprocessing`) |
| `k_means_plus_plus_initialization` | scikit-learn examples/cluster/plot_kmeans_plusplus.py · "An example of K-Means++ initialization" | available now (`sciencekit_cluster`) |
| `k_means_versus_mini_batch_k_means_on_a_stream` | scikit-learn examples/cluster/plot_mini_batch_kmeans.py · "Comparison of the K-Means and MiniBatchKMeans clustering algorithms" | available now (`sciencekit_cluster`, `sciencekit_common` streaming contracts) |
| `dbscan_clustering_of_non_convex_shapes` | scikit-learn examples/cluster/plot_dbscan.py · "Demo of DBSCAN clustering algorithm" | available now (`sciencekit_cluster`) |
| `density_reachability_of_optics` | scikit-learn examples/cluster/plot_optics.py · "Demo of OPTICS clustering algorithm" | planned (`sciencekit_cluster` — the OPTICS variant on top of the implemented DBSCAN) |
| `choosing_the_number_of_clusters_with_a_silhouette_analysis` | scikit-learn examples/cluster/plot_kmeans_silhouette_analysis.py · "Selecting the number of clusters with silhouette analysis on KMeans clustering" | available now (`sciencekit_cluster`, `sciencekit_metrics` clustering scorers) |
| `clustering_under_a_feature_scale_mismatch` | no scikit-learn origin — a sciencekit-native extension · "Why k-means needs scaling" | available now (`sciencekit_cluster`, `sciencekit_preprocessing`) |
| `hierarchical_clustering_of_toy_datasets` | scikit-learn examples/cluster/plot_linkage_comparison.py · "Comparing different hierarchical linkage methods on toy datasets" | planned (`sciencekit_cluster` — the agglomerative family) |
| `density_based_clustering_of_anisotropic_shapes` | scikit-learn examples/cluster/plot_cluster_comparison.py · "Comparing different clustering algorithms on toy datasets" | available now (`sciencekit_cluster`) |
| `spectral_clustering_for_image_segmentation` | scikit-learn examples/cluster/plot_segmentation_toy.py · "Spectral clustering for image segmentation" | planned (`sciencekit_cluster` — the spectral family) |
| `bisecting_k_means_against_regular_k_means` | scikit-learn examples/cluster/plot_bisect_kmeans.py · "Bisecting K-Means and Regular K-Means Performance Comparison" | planned (`sciencekit_cluster`) |
| `clustering_of_sparse_documents` | scikit-learn examples/cluster/plot_coin_ward_segmentation.py · "Segmenting the picture of greek coins in regions" | available now (`sciencekit_cluster`, `sciencekit_math` sparse operations) |
| `principal_component_analysis_on_the_iris_dataset` | scikit-learn examples/decomposition/plot_pca_iris.py · "Principal Component Analysis (PCA) on Iris Dataset" | available now (`sciencekit_decomposition`, `sciencekit_preprocessing`) |
| `choosing_the_number_of_components_by_the_variance_spectrum` | scikit-learn examples/decomposition/plot_pca_vs_fa_model_selection.py · "Model selection with Probabilistic PCA and Factor Analysis (FA)" | available now (`sciencekit_decomposition`) |
| `incremental_pca_on_a_stream` | scikit-learn examples/decomposition/plot_incremental_pca.py · "Incremental PCA" | available now (`sciencekit_decomposition`, `sciencekit_common` streaming contracts) |
| `truncated_svd_on_a_sparse_document_design` | scikit-learn examples/decomposition/plot_faces_decomposition.py · "Faces dataset decompositions" (sparse arm) | | available now (`sciencekit_decomposition`) |
| `randomized_svd_for_large_low_rank_designs` | no scikit-learn origin — a sciencekit-native extension · "Randomized SVD" | available now (`sciencekit_decomposition`) |
| `projection_of_a_high_dimensional_space_onto_two_components` | scikit-learn examples/decomposition/plot_pca_iris.py · "PCA: 2D projection of a 4D dataset" | available now (`sciencekit_decomposition`) |
| `kernel_principal_component_analysis_of_a_nonlinear_structure` | scikit-learn examples/decomposition/plot_kernel_pca.py · "Kernel PCA" | planned (`sciencekit_decomposition` — the kernel PCA on top of the implemented kernels) |
| `factor_analysis_with_rotation` | scikit-learn examples/decomposition/plot_varimax_fa.py · "Factor Analysis (with rotation) to visualize patterns" | planned (`sciencekit_decomposition` — the factor-analysis family) |
| `blind_source_separation_with_fast_ica` | scikit-learn examples/decomposition/plot_ica_blind_source_separation.py · "Blind source separation using FastICA" | planned (`sciencekit_decomposition` — the ICA family) |
| `sparse_coding_with_a_precomputed_dictionary` | scikit-learn examples/decomposition/plot_sparse_coding.py · "Sparse coding with a precomputed dictionary" | planned (`sciencekit_decomposition` — the sparse-coding and dictionary-learning family) |
| `non_negative_matrix_factorization_of_a_document_term_matrix` | scikit-learn examples/applications/plot_topics_extraction_with_nmf_lda.py · "Topic extraction with Non-negative Matrix Factorization" | | planned (`sciencekit_decomposition` — the NMF family) |
| `manifold_learning_of_a_swiss_roll` | scikit-learn examples/manifold/plot_swissroll.py · "Swiss Roll And Swiss-Hole Reduction" | absent (no sciencekit path yet — manifold learning is not on the roadmap) |
| `kernel_approximation_of_an_explicit_feature_map` | scikit-learn examples/miscellaneous/plot_kernel_approximation.py · "Explicit feature map approximation for RBF kernels" | planned (`sciencekit_decomposition` — the kernel-approximation family) |
| `cross_validated_score_of_a_classifier` | scikit-learn examples/model_selection/plot_cross_val_score.py · "Cross-validation" | available now (`sciencekit_model_selection`) |
| `balancing_model_complexity_and_the_cross_validated_score` | scikit-learn examples/model_selection/plot_grid_search_refit_callable.py · "Balance model complexity and cross-validated score" | available now (`sciencekit_model_selection`) |
| `comparing_randomized_search_and_grid_search` | scikit-learn examples/model_selection/plot_randomized_search.py · "Comparing randomized search and grid search for hyperparameter estimation" | planned (`sciencekit_model_selection` — the randomized search) |
| `comparison_between_grid_search_and_successive_halving` | scikit-learn examples/model_selection/plot_successive_halving_heatmap.py · "Comparison between grid search and successive halving" | available now (`sciencekit_model_selection` — the halving search) |
| `learning_curve_of_a_classifier` | scikit-learn examples/model_selection/plot_learning_curve.html · "Learning curve" | planned (`sciencekit_model_selection` — the learning-curve utility) |
| `validation_curve_of_a_regularization_strength` | scikit-learn examples/model_selection/plot_validation_curve.html · "Validation curve" | planned (`sciencekit_model_selection` — the validation-curve utility) |
| `pipeline_of_scaling_and_a_linear_model` | scikit-learn examples/compose/plot_pipeline_display.py · "Displaying Pipelines" | | available now (`sciencekit_pipeline`) |
| `column_transformer_over_mixed_column_types` | scikit-learn examples/compose/plot_column_transformer.py · "Mixed-type columns in a ColumnTransformer" | available now (`sciencekit_pipeline`, `sciencekit_interop` column selection) |
| `feature_union_of_two_views_of_the_same_data` | scikit-learn examples/compose/plot_feature_union.py · "Feature Union" | available now (`sciencekit_pipeline`) |
| `streamed_pipeline_over_a_live_source` | no scikit-learn origin — a sciencekit-native extension · "A pipeline that never holds the whole dataset" | available now (`sciencekit_pipeline` streaming contracts, `sciencekit_common` batching) |
| `custom_refit_strategy_of_a_grid_search` | scikit-learn examples/model_selection/plot_grid_search_digits.py · "Custom refit strategy of a grid search with cross-validation" | planned (`sciencekit_model_selection` — the per-fold refit strategy) |
| `confusion_matrix_of_a_classifier` | scikit-learn examples/model_selection/plot_confusion_matrix.py · "Evaluate the performance of a classifier with Confusion Matrix" | available now (`sciencekit_metrics`, `sciencekit_neighbors`) |
| `roc_curve_of_a_binary_classifier` | scikit-learn examples/model_selection/plot_roc_curve.html · "Receiver Operating Characteristic (ROC)" | available now (`sciencekit_metrics` ranking and probability scorers) |
| `precision_recall_curve_under_class_imbalance` | scikit-learn examples/model_selection/plot_precision_recall_curve.html · "Precision-Recall curve" | available now (`sciencekit_metrics` ranking scorers) |
| `detection_error_tradeoff_curve` | scikit-learn examples/model_selection/plot_det.html · "Detection error tradeoff (DET) curve" | planned (`sciencekit_metrics` — the DET utility on top of the existing ranking scorers) |
| `multi_metric_evaluation_of_a_cross_validated_score` | scikit-learn examples/model_selection/plot_multi_metric_evaluation.py · "Demonstration of multi-metric evaluation on cross_val_score and GridSearchCV" | available now (`sciencekit_metrics` scorers, `sciencekit_model_selection`) |
| `class_likelihood_ratios_for_a_classifier` | scikit-learn examples/model_selection/plot_likelihood_ratios.html · "Class Likelihood Ratios to measure classification performance" | planned (`sciencekit_metrics` — the likelihood-ratio utility) |
| `permutation_importance_against_forest_importance` | scikit-learn examples/inspection/plot_permutation_importance.py · "Permutation Importance vs Random Forest Feature Importance (MDI)" | planned (`sciencekit_metrics` — the permutation importance) |
| `permutation_importance_with_correlated_features` | scikit-learn examples/inspection/plot_permutation_importance_multicollinear.py · "Permutation Importance with Multicollinear or Correlated Features" | planned (`sciencekit_metrics` — the permutation importance) |
| `partial_dependence_of_a_model` | scikit-learn examples/inspection/plot_partial_dependence.py · "Partial Dependence and Individual Conditional Expectation Plots" | planned (`sciencekit_metrics` — the partial-dependence utility) |
| `probability_calibration_curve_of_a_classifier` | scikit-learn examples/calibration/plot_calibration_curve.py · "Probability Calibration curves" | planned (`sciencekit_metrics` — the calibration metrics are planned with the calibration and multilabel metrics change) |
| `classification_report_of_a_multiclass_model` | scikit-learn examples/classification/plot_classification_probability.py · "Plot classification probability" | available now (`sciencekit_metrics`, `sciencekit_neighbors`) |
| `univariate_feature_selection_by_a_statistical_test` | scikit-learn examples/feature_selection/plot_feature_selection.py · "Univariate Feature Selection" | planned (`sciencekit_feature_selection` — the family is not implemented yet) |
| `recursive_feature_elimination` | scikit-learn examples/feature_selection/plot_rfe_digits.py · "Recursive Feature Elimination" | planned (`sciencekit_feature_selection`) |
| `model_based_and_sequential_feature_selection` | scikit-learn examples/feature_selection/plot_select_from_model_diabetes.py · "Model-based and Sequential Feature Selection" | planned (`sciencekit_feature_selection`) |
| `feature_selection_inside_a_pipeline` | scikit-learn examples/feature_selection/plot_feature_selection_pipeline.py · "Pipeline ANOVA SVM" | planned (`sciencekit_feature_selection`, `sciencekit_model_selection`) |
| `feature_agglomeration_against_univariate_selection` | scikit-learn examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py · "Feature agglomeration vs. univariate selection" | planned (`sciencekit_feature_selection`) |
| `cyclic_encoding_of_a_time_feature` | scikit-learn examples/applications/plot_cyclical_feature_engineering.py · "Time-related feature engineering" | planned (`sciencekit_examples` — the encoding is an example-local feature-engineering step, not a library transform) |
| `lagged_features_for_a_time_series` | scikit-learn examples/applications/plot_time_series_lagged_features.py · "Lagged features for time series forecasting" | planned (`sciencekit_examples` — the lag construction is example-local) |
| `one_hot_encoding_of_a_low_cardinality_feature` | scikit-learn examples/preprocessing/plot_one_hot_encoder.py · "One-hot encoding of categorical features" | | available now (`sciencekit_preprocessing`) |
| `interaction_features_between_categorical_and_numeric_columns` | no scikit-learn origin — a sciencekit-native extension · "Cross-type interactions" | planned (`sciencekit_examples` — the interaction construction is example-local) |
| `target_encoding_of_a_high_cardinality_feature` | no scikit-learn origin — a sciencekit-native extension · "Target encoding, and the leakage it invites" | planned (`sciencekit_examples` — the encoding is example-local, and the leakage demonstration is its point) |
| `variance_threshold_as_a_zero_variance_filter` | no scikit-learn origin — a sciencekit-native extension · "Dropping the constants" | planned (`sciencekit_examples` — the filter is a column selection over the fitted design) |
| `isolation_forest_as_an_outlier_detector` | scikit-learn examples/ensemble/plot_isolation_forest.py · "IsolationForest example" | available now (`sciencekit_outlier`) |
| `comparing_anomaly_detection_estimators` | scikit-learn examples/miscellaneous/plot_anomaly_comparison.py · "Comparing anomaly detection algorithms for outlier detection on toy datasets" | planned (`sciencekit_outlier` — the local-outlier factor and the one-class SVM are tracked in the outlier LOF and one-class SVM change) |
| `outlier_detection_on_a_real_dataset` | scikit-learn examples/applications/plot_outlier_detection_wine.py · "Outlier detection on a real data set" | available now (`sciencekit_outlier`, `sciencekit_preprocessing`) |
| `empirical_against_shrunk_covariance` | scikit-learn examples/covariance/plot_covariance_estimation.py · "Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood" | absent (no sciencekit path yet — the covariance family is not on the roadmap) |
| `mahalanobis_distances_under_a_robust_covariance` | scikit-learn examples/covariance/plot_mahalanobis_distances.py · "Robust covariance estimation and Mahalanobis distances relevance" | absent (no sciencekit path yet) |
| `gaussian_mixture_model_of_a_density` | scikit-learn examples/mixture/plot_gmm_pdf.py · "Density Estimation for a Gaussian mixture" | absent (no sciencekit path yet — the mixture family is not on the roadmap) |
| `gaussian_process_regression_of_a_noisy_function` | scikit-learn examples/gaussian_process/plot_gpr_noisy_targets.py · "Gaussian Processes regression: basic introductory example" | absent (no sciencekit path yet) |
| `neural_network_classifier_on_a_synthetic_boundary` | scikit-learn examples/neural_networks/plot_mlp_classifier.py · "Plot a decision function for a multilayer perceptron" | absent (no sciencekit path yet — the neural-network family is not on the roadmap) |
| `text_documents_as_a_sparse_design` | scikit-learn examples/text/plot_document_classification_20newsgroups.py · "Text classification" | planned (`sciencekit_examples` — the vectorizers are example-local; the estimator is available now) |
| `naive_bayes_for_text_categorical_data` | scikit-learn examples/text/plot_document_classification_20newsgroups.py · "Naive Bayes" arm | | absent (no sciencekit path yet — the naive Bayes family is not on the roadmap) |
| `semi_supervised_learning_with_labeled_and_unlabeled_samples` | scikit-learn examples/semi_supervised/plot_label_spreading.py · "Label Spreading" | absent (no sciencekit path yet) |
| `multiclass_strategy_for_a_binary_only_estimator` | scikit-learn examples/multiclass/plot_multiclass.py · "Multiclass classification" | planned (`sciencekit_common` — the multiclass strategy on top of the existing predictor contracts) |
| `multioutput_regression_with_shared_structure` | scikit-learn examples/multioutput/plot_multioutput_face_completion.py · "Face completion with a multi-output estimator" | absent (no sciencekit path yet) |
| `cross_decomposition_of_two_views` | scikit-learn examples/cross_decomposition/plot_compare_cross_decomposition.py · "Compare cross decomposition methods" | absent (no sciencekit path yet) |
| `biclustering_of_a_block_structure` | scikit-learn examples/bicluster/plot_spectral_coclustering.py · "A demo of the Spectral Co-Clustering algorithm" | absent (no sciencekit path yet) |
| `kernel_approximation_for_scalable_learning` | scikit-learn examples/kernel_approximation/plot_scalable_poly_kernels.py · "Scalable learning with polynomial kernel approximation" | planned (`sciencekit_decomposition` — the kernel-approximation family) |
| `calibration_of_a_classifier_under_distribution_shift` | scikit-learn examples/calibration/plot_compare_calibration.py · "Comparison of Calibration of Classifiers" | planned (`sciencekit_metrics` — the calibration metrics are planned with the calibration and multilabel metrics change) |
| `out_of_core_classification_of_a_stream` | scikit-learn examples/applications/plot_out_of_core_classification.py · "Out-of-core classification of text documents" | available now (`sciencekit_common` streaming contracts, `sciencekit_linear_model`) |
| `outlier_detection_in_a_feature_space_with_mixed_scales` | no scikit-learn origin — a sciencekit-native extension · "Scale and the anomaly score" | available now (`sciencekit_outlier`, `sciencekit_preprocessing`) |

## The crates the corpus leans on

| Crate | Examples available now |
|---|---|
| `sciencekit_preprocessing` | 17 |
| `sciencekit_linear_model` | 15 |
| `sciencekit_ensemble` | 9 |
| `sciencekit_metrics` | 8 |
| `sciencekit_cluster` | 8 |
| `sciencekit_neighbors` | 7 |
| `sciencekit_common` | 6 |
| `sciencekit_decomposition` | 6 |
| `sciencekit_model_selection` | 5 |
| `sciencekit_pipeline` | 5 |
| `sciencekit_impute` | 4 |
| `sciencekit_svm` | 4 |
| `sciencekit_outlier` | 3 |
| `sciencekit_math` | 2 |
| `sciencekit_tree` | 2 |
| `sciencekit_interop` | 1 |

## The families sciencekit does not cover

These are the deliberate omissions of the corpus. Each one is a *decision*
recorded in the catalog with its reason, not an oversight:

- `theil_sen_regression_against_ransac` — scikit-learn `examples/linear_model/plot_theilsen.py` · "Theil-Sen Regression"
- `manifold_learning_of_a_swiss_roll` — scikit-learn `examples/manifold/plot_swissroll.py` · "Swiss Roll And Swiss-Hole Reduction"
- `empirical_against_shrunk_covariance` — scikit-learn `examples/covariance/plot_covariance_estimation.py` · "Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood"
- `mahalanobis_distances_under_a_robust_covariance` — scikit-learn `examples/covariance/plot_mahalanobis_distances.py` · "Robust covariance estimation and Mahalanobis distances relevance"
- `gaussian_mixture_model_of_a_density` — scikit-learn `examples/mixture/plot_gmm_pdf.py` · "Density Estimation for a Gaussian mixture"
- `gaussian_process_regression_of_a_noisy_function` — scikit-learn `examples/gaussian_process/plot_gpr_noisy_targets.py` · "Gaussian Processes regression: basic introductory example"
- `neural_network_classifier_on_a_synthetic_boundary` — scikit-learn `examples/neural_networks/plot_mlp_classifier.py` · "Plot a decision function for a multilayer perceptron"
- `naive_bayes_for_text_categorical_data` — scikit-learn `examples/text/plot_document_classification_20newsgroups.py` · "Naive Bayes" arm |
- `semi_supervised_learning_with_labeled_and_unlabeled_samples` — scikit-learn `examples/semi_supervised/plot_label_spreading.py` · "Label Spreading"
- `multioutput_regression_with_shared_structure` — scikit-learn `examples/multioutput/plot_multioutput_face_completion.py` · "Face completion with a multi-output estimator"
- `cross_decomposition_of_two_views` — scikit-learn `examples/cross_decomposition/plot_compare_cross_decomposition.py` · "Compare cross decomposition methods"
- `biclustering_of_a_block_structure` — scikit-learn `examples/bicluster/plot_spectral_coclustering.py` · "A demo of the Spectral Co-Clustering algorithm"

## The capabilities the corpus needs next

| Crate | Examples waiting on it |
|---|---|
| `sciencekit_metrics` | 8 |
| `sciencekit_decomposition` | 7 |
| `sciencekit_examples` | 6 |
| `sciencekit_model_selection` | 5 |
| `sciencekit_feature_selection` | 5 |
| `sciencekit_cluster` | 4 |
| `sciencekit_linear_model` | 2 |
| `sciencekit_outlier` | 2 |
| `sciencekit_ensemble` | 2 |
| `sciencekit_tree` | 1 |
| `sciencekit_common` | 1 |

