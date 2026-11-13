//! Companion validation tests for `SKGradientBoostingClassifier` and
//! `SKGradientBoostingRegressor` asserting exact central `SKError` variants
//! (task 6.3).

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKExecutionMode, SKRegressorPredictor,
    SKSupervisedFit,
};
use sciencekit_tree::SKMaximumFeatures;
use sprs::CsMat;

use super::{SKGradientBoostingClassifier, SKGradientBoostingRegressor};

fn default_classifier() -> SKGradientBoostingClassifier<f64> {
    SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(5)
        .build()
        .expect("builds")
}
fn default_regressor() -> SKGradientBoostingRegressor<f64> {
    SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(5)
        .build()
        .expect("builds")
}

/// Empty input, zero columns, target-length and feature-count mismatches fail.
#[test]
fn empty_input_and_shape_mismatches_are_rejected() {
    let empty_dense: Array2<f64> = Array2::zeros((0, 2));
    let empty_sparse: CsMat<f64> = CsMat::empty(sprs::CSR, 2);
    let (labels, responses) = (array![0_i64, 1], array![1.0, 2.0]);
    for outcome in [
        default_classifier()
            .fit(&empty_dense, labels.view())
            .map(|_| ()),
        default_classifier()
            .fit(empty_sparse.view(), labels.view())
            .map(|_| ()),
        default_regressor()
            .fit(&empty_dense, responses.view())
            .map(|_| ()),
    ] {
        assert!(matches!(
            outcome,
            Err(SKError::EmptyInput { operation: "fit" })
        ));
    }
    assert!(matches!(
        default_classifier().fit(&Array2::<f64>::zeros((2, 0)), labels.view()),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        default_regressor().fit(&array![[0.0], [1.0], [2.0]], responses.view()),
        Err(SKError::ShapeMismatch { .. })
    ));

    let training = array![[0.0, 1.0], [2.0, 3.0]];
    let fitted_classifier = default_classifier()
        .fit(&training, labels.view())
        .expect("fit");
    let fitted_regressor = default_regressor()
        .fit(&training, responses.view())
        .expect("fit");
    assert!(matches!(
        fitted_classifier.predict_labels(&empty_dense),
        Err(SKError::EmptyInput {
            operation: "predict"
        })
    ));
    assert!(matches!(
        fitted_regressor.predict(&empty_dense),
        Err(SKError::EmptyInput {
            operation: "predict"
        })
    ));
    assert!(matches!(
        fitted_classifier.predict_probabilities(array![[1.0, 2.0, 3.0]].view()),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        fitted_regressor.predict(array![[1.0]].view()),
        Err(SKError::ShapeMismatch { .. })
    ));
}

fn assert_invalid_hyperparameter<T: std::fmt::Debug>(
    outcome: Result<T, SKError>,
    expected: &'static str,
) {
    match outcome {
        Err(SKError::InvalidHyperparameter { name, .. }) => assert_eq!(name, expected),
        other => panic!("expected InvalidHyperparameter({expected}), got {other:?}"),
    }
}

/// Scenarios: Non-positive learning rate/estimator count, invalid
/// subsample/feature/regularization/validation-fraction parameters, early-stop
/// split exhaustion and clearing early stopping via `without_early_stopping`.
#[test]
fn hyperparameter_bounds_and_early_stop_validation_are_enforced() {
    let classifier = || SKGradientBoostingClassifier::<f64>::build();
    let regressor = || SKGradientBoostingRegressor::<f64>::build();
    assert_eq!(
        classifier()
            .n_iter_no_change(3)
            .without_early_stopping()
            .build()
            .expect("ok")
            .n_iter_no_change(),
        None
    );
    assert_eq!(
        regressor()
            .n_iter_no_change(3)
            .without_early_stopping()
            .build()
            .expect("ok")
            .n_iter_no_change(),
        None
    );
    assert_invalid_hyperparameter(classifier().learning_rate(0.0).build(), "learning_rate");
    assert_invalid_hyperparameter(
        regressor().number_of_estimators(0).build(),
        "number_of_estimators",
    );
    assert_invalid_hyperparameter(classifier().subsample(1.5).build(), "subsample");
    assert_invalid_hyperparameter(
        regressor()
            .maximum_features(SKMaximumFeatures::Fraction(0.0))
            .build(),
        "maximum_features",
    );
    assert_invalid_hyperparameter(
        classifier().l2_regularization(-1.0).build(),
        "l2_regularization",
    );
    assert_invalid_hyperparameter(
        regressor().validation_fraction(1.0).build(),
        "validation_fraction",
    );
    assert_invalid_hyperparameter(
        regressor()
            .n_iter_no_change(2)
            .build()
            .expect("builds")
            .fit(&array![[1.0]], array![2.0].view()),
        "validation_fraction",
    );
}

/// Scenarios: Single-class classification, non-finite values at exact positions,
/// wrong target kinds and incompatible streaming execution modes are rejected.
#[test]
fn single_class_non_finite_values_and_streaming_modes_are_rejected() {
    match default_classifier().fit(&array![[0.0], [1.0], [2.0]], array![7_i64, 7, 7].view()) {
        Err(SKError::InvalidHyperparameter { name, reason }) => {
            assert_eq!(name, "targets");
            assert!(reason.contains("two classes"));
        }
        other => panic!("expected InvalidHyperparameter(targets), got {other:?}"),
    }

    let bad_sparse = CsMat::new((2, 2), vec![0, 1, 2], vec![1, 0], vec![f64::NAN, 1.0]);
    let bad_dense = array![[0.0, f64::NAN], [1.0, 2.0]];
    let clean = array![[0.0, 1.0], [2.0, 3.0]];
    let fitted = default_regressor()
        .fit(&clean, array![1.0, 2.0].view())
        .expect("fit");
    for non_finite in [
        default_classifier()
            .fit(&bad_dense, array![0_i64, 1].view())
            .map(|_| ()),
        default_classifier()
            .fit(bad_sparse.view(), array![0_i64, 1].view())
            .map(|_| ()),
        default_regressor()
            .fit(&clean, array![1.0, f64::INFINITY].view())
            .map(|_| ()),
        fitted.predict(bad_sparse.view()).map(|_| ()),
    ] {
        assert!(matches!(
            non_finite,
            Err(SKError::NonFiniteValue { position: 1 })
        ));
    }
    assert!(matches!(
        default_classifier().fit(&clean, array![0.1, 0.9].view()),
        Err(SKError::UnsupportedRepresentation { .. })
    ));
    assert!(matches!(
        SKGradientBoostingRegressor::<f64>::build()
            .execution_mode(SKExecutionMode::OutOfCoreStreaming)
            .build()
            .expect("builds")
            .fit(&clean, array![1.0, 2.0].view()),
        Err(SKError::ExecutionModeIncompatible { .. })
    ));
}
