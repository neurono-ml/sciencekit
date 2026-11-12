//! Tiny-data acceptance tests for generic bagging: `f32`/`f64`, single-class
//! bootstrap resilience, tree and non-tree base estimators, feature bootstrap,
//! scorer adapters and `SKExportable` state export.

use ndarray::array;
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKExportable, SKLabelScorer, SKRegressorPredictor,
    SKSupervisedFit, SKSupervisedScorer,
};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_metrics::scorers::{SKAccuracyScorer, SKR2Scorer};
use sciencekit_neighbors::SKKNeighborsClassifier;
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};

use crate::{SKBaggingClassifier, SKBaggingRegressor};

#[test]
fn tiny_tree_and_non_tree_bagging_fit_score_and_export_in_f32_and_f64() {
    let features_f64 = array![
        [0.0, 0.1, 0.2],
        [0.1, 0.2, 0.3],
        [5.0, 5.1, 5.2],
        [5.1, 5.2, 5.3]
    ];
    let labels = array![0_i64, 0, 1, 1];

    // 40 estimators on 4 rows routinely produce single-class bootstrap draws;
    // both tree and logistic-regression bases must succeed and score cleanly.
    let tree_base = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    let tree_bagging = SKBaggingClassifier::build(tree_base)
        .number_of_estimators(40)
        .maximum_features(SKMaximumFeatures::Count(2))
        .bootstrap_features(true)
        .oob_score(true)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(&features_f64, labels.view())
        .unwrap();

    let tree_accuracy = SKAccuracyScorer
        .score(
            &tree_bagging,
            features_f64.view().try_into().unwrap(),
            labels.view().try_into().unwrap(),
        )
        .unwrap();
    assert!((tree_accuracy - 1.0).abs() < 1e-12);
    assert!(tree_bagging.oob_scored_rows() > 0);
    assert!(tree_bagging.oob_score().unwrap() >= 0.0);

    let logistic_base = SKLogisticRegression::<f64>::build().build().unwrap();
    let logistic_bagging = SKBaggingClassifier::build(logistic_base)
        .number_of_estimators(25)
        .oob_score(true)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(&features_f64, labels.view())
        .unwrap();
    assert_eq!(
        logistic_bagging.predict_labels(&features_f64).unwrap(),
        labels
    );

    // f32 non-tree classifier and tree regressor.
    let features_f32 = array![
        [0.0_f32, 0.1],
        [0.2, 0.3],
        [8.0, 8.1],
        [8.2, 8.3],
        [16.0, 16.1],
        [16.2, 16.3]
    ];
    let labels_f32 = array![0_i64, 0, 1, 1, 2, 2];
    let knn_f32 = SKKNeighborsClassifier::<f32>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap();
    let knn_bagging = SKBaggingClassifier::build(knn_f32)
        .number_of_estimators(8)
        .random_seed(3)
        .build()
        .unwrap()
        .fit(&features_f32, labels_f32.view())
        .unwrap();
    assert_eq!(knn_bagging.predict_labels(&features_f32).unwrap().len(), 6);

    let responses_f32 = array![1.0_f32, 1.2, 8.0, 8.2, 16.0, 16.2];
    let tree_regressor_f32 = SKDecisionTreeRegressor::<f32>::build().build().unwrap();
    let regressor_bagging_f32 = SKBaggingRegressor::build(tree_regressor_f32)
        .number_of_estimators(20)
        .oob_score(true)
        .random_seed(11)
        .build()
        .unwrap()
        .fit(&features_f32, responses_f32.view())
        .unwrap();
    let r2_f32 = SKR2Scorer
        .score(
            &regressor_bagging_f32,
            features_f32.view().try_into().unwrap(),
            responses_f32.view().try_into().unwrap(),
        )
        .unwrap();
    assert!(r2_f32 > 0.85, "f32 R² {r2_f32}");

    // Export model state when base model implements SKExportable (SKLinearRegression).
    let regression_features = array![
        [1.0_f64, 0.0],
        [0.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [3.0, 2.0],
        [2.0, 3.0]
    ];
    let regression_targets = array![3.0_f64, 5.0, 9.0, 11.0, 15.0, 17.0];
    let linear_bagging = SKBaggingRegressor::build(SKLinearRegression::build().build().unwrap())
        .number_of_estimators(4)
        .maximum_features(SKMaximumFeatures::Count(2))
        .oob_score(true)
        .random_seed(19)
        .build()
        .unwrap()
        .fit(&regression_features, regression_targets.view())
        .unwrap();
    assert_eq!(
        linear_bagging.predict(&regression_features).unwrap().len(),
        6
    );

    let exported = linear_bagging.export_state().unwrap();
    assert_eq!(exported.metadata().estimator_name(), "SKBaggingRegressor");
    assert_eq!(exported.metadata().feature_count(), 2);
    assert!(exported.tensor("bagging.feature_subsets").is_some());
    assert!(exported.tensor("estimator.0.linear.coefficients").is_some());
}
