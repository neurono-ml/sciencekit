//! Validation tests for the histogram boosters (task 5.2).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKError, SKRegressorPredictor, SKSupervisedFit};

use super::SKHistGradientBoostingRegressor;

fn regressor() -> super::SKHistGradientBoostingRegressor<f64> {
    SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(2)
        .without_early_stopping()
        .minimum_samples_leaf(1)
        .build()
        .expect("build")
}

#[test]
fn hist_validation_rejects_empty_and_non_finite_input() {
    let empty = Array2::<f64>::zeros((0, 2));
    let targets = ndarray::Array1::<f64>::zeros(0);
    assert!(matches!(
        regressor().fit(empty.view(), targets.view()),
        Err(SKError::EmptyInput { .. })
    ));
    let features = Array2::from_shape_vec((3, 1), vec![0.0, f64::INFINITY, 2.0]).expect("data");
    let responses = ndarray::Array1::from_vec(vec![0.0, 1.0, 2.0]);
    assert!(matches!(
        regressor().fit(features.view(), responses.view()),
        Err(SKError::NonFiniteValue { .. })
    ));
}

#[test]
fn hist_validation_accepts_missing_values_and_rejects_unknown_categorical_columns() {
    let features =
        Array2::from_shape_vec((4, 2), vec![0.0, 1.0, f64::NAN, 2.0, 2.0, 3.0, 3.0, 4.0])
            .expect("data");
    let responses = ndarray::Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    assert!(regressor().fit(features.view(), responses.view()).is_ok());
    let error = SKHistGradientBoostingRegressor::<f64>::build()
        .categorical_features(vec![5])
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), responses.view())
        .expect_err("unknown categorical column must fail");
    match error {
        SKError::InvalidHyperparameter { name, reason } => {
            assert_eq!(name, "categorical_features");
            assert!(reason.contains('5'));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn hist_validation_rejects_feature_count_mismatch_and_invalid_fraction() {
    let features = Array2::from_shape_vec((4, 2), vec![0.0; 8]).expect("data");
    let responses = ndarray::Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let model = regressor()
        .fit(features.view(), responses.view())
        .expect("fit");
    let wrong = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("data");
    assert!(matches!(
        model.predict(wrong.view()),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        SKHistGradientBoostingRegressor::<f64>::build()
            .validation_fraction(1.0)
            .build(),
        Err(SKError::InvalidHyperparameter {
            name: "validation_fraction",
            ..
        })
    ));
    assert!(matches!(
        SKHistGradientBoostingRegressor::<f64>::build()
            .minimum_samples_leaf(0)
            .build(),
        Err(SKError::InvalidHyperparameter { .. })
    ));
    assert!(matches!(
        SKHistGradientBoostingRegressor::<f64>::build()
            .l2_regularization(-1.0)
            .build(),
        Err(SKError::InvalidHyperparameter { .. })
    ));
}

#[test]
fn hist_validation_rejects_csc_and_non_finite_targets() {
    let dense =
        Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0]).expect("data");
    let csc = sprs::CsMat::new_csc(
        (4, 2),
        vec![0, 2, 4],
        vec![0, 2, 1, 3],
        vec![1.0, 3.0, 2.0, 4.0],
    );
    let responses = ndarray::Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    assert!(matches!(
        regressor().fit(csc.view(), responses.view()),
        Err(SKError::UnsupportedRepresentation { .. })
    ));
    let broken_targets = ndarray::Array1::from_vec(vec![0.0, f64::NAN, 2.0, 3.0]);
    assert!(matches!(
        regressor().fit(dense.view(), broken_targets.view()),
        Err(SKError::NonFiniteValue { .. })
    ));
}
