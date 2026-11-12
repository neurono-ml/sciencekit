//! Histogram classifier builder and prediction tests (task 4.2).

use ndarray::Array2;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKError, SKExecutionMode, SKSupervisedFit,
};

use super::SKHistGradientBoostingClassifier;

fn two_class_data() -> (Array2<f64>, ndarray::Array1<i64>) {
    let rows = 120usize;
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| {
        let side = if row % 2 == 0 { 0.0 } else { 3.0 };
        side + (row / 2 % 7) as f64 * 0.05 + column as f64 * 0.01
    });
    let labels = ndarray::Array1::from_shape_fn(rows, |row| (row % 2) as i64);
    (features, labels)
}

#[test]
fn hist_classifier_defaults_match_the_documented_configuration() {
    let estimator = SKHistGradientBoostingClassifier::<f64>::build()
        .build()
        .expect("defaults");
    assert_eq!(estimator.number_of_estimators(), 100);
    assert_eq!(estimator.maximum_bins(), 255);
    assert_eq!(estimator.maximum_leaf_nodes(), 31);
    assert_eq!(estimator.minimum_samples_leaf(), 20);
    assert!(estimator.categorical_features().is_empty());
    assert_eq!(estimator.n_iter_no_change(), Some(10));
    assert_eq!(estimator.learning_rate(), 0.1);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
}

#[test]
fn hist_classifier_separates_a_binary_problem() {
    let (features, labels) = two_class_data();
    let model = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(20)
        .learning_rate(0.3)
        .maximum_leaf_nodes(4)
        .minimum_samples_leaf(1)
        .maximum_bins(16)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("probabilities");
    assert_eq!(probabilities.dim(), (120, 2));
    let predicted = model.predict_labels(features.view()).expect("labels");
    let correct = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(predicted, actual)| predicted == actual)
        .count();
    assert!(correct >= 115, "correct = {correct}");
    assert!(model.number_of_stages() > 0);
    assert!(model.bin_edges(0).is_some());
    assert!(model.feature_importances().iter().sum::<f64>() > 0.0);
}

#[test]
fn hist_classifier_handles_three_classes() {
    let rows = 150usize;
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| {
        (row % 3) as f64 + column as f64 * 0.1
    });
    let labels = ndarray::Array1::from_shape_fn(rows, |row| (row % 3) as i64);
    let model = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(10)
        .learning_rate(0.3)
        .maximum_leaf_nodes(4)
        .minimum_samples_leaf(1)
        .maximum_bins(8)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    assert_eq!(model.number_of_classes(), 3);
    assert_eq!(model.trees_per_stage(), 3);
    assert_eq!(
        model
            .predict_probabilities(features.view())
            .expect("probabilities")
            .dim(),
        (150, 3)
    );
}

#[test]
fn hist_classifier_rejects_invalid_parameters_and_single_class_targets() {
    assert!(matches!(
        SKHistGradientBoostingClassifier::<f64>::build()
            .maximum_bins(0)
            .build(),
        Err(SKError::InvalidHyperparameter {
            name: "maximum_bins",
            ..
        })
    ));
    assert!(matches!(
        SKHistGradientBoostingClassifier::<f64>::build()
            .maximum_bins(256)
            .build(),
        Err(SKError::InvalidHyperparameter { .. })
    ));
    assert!(matches!(
        SKHistGradientBoostingClassifier::<f64>::build()
            .maximum_leaf_nodes(1)
            .build(),
        Err(SKError::InvalidHyperparameter { .. })
    ));
    assert!(matches!(
        SKHistGradientBoostingClassifier::<f64>::build()
            .learning_rate(0.0)
            .build(),
        Err(SKError::InvalidHyperparameter { .. })
    ));
    let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).expect("data");
    let labels = ndarray::Array1::from_elem(6, 1i64);
    let error = SKHistGradientBoostingClassifier::<f64>::build()
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect_err("single class must fail");
    assert!(matches!(error, SKError::InvalidHyperparameter { name: "targets", .. }));
}
