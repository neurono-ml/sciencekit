//! Regressor fit and prediction tests (tasks 5.2 and 5.3).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKRegressorPredictor, SKSupervisedFit};
use sprs::CsMat;

use crate::criterion::SKTreeCriterion;

use super::SKDecisionTreeRegressor;

/// With squared error, a single leaf predicts the mean of its responses.
#[test]
fn squared_error_predicts_the_leaf_mean() {
    let features = array![[0.0], [1.0], [2.0]];
    let responses = array![1.0, 2.0, 6.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(
        model.predict(features.view()).expect("predict"),
        array![3.0, 3.0, 3.0]
    );
}

/// With absolute error, a single leaf predicts the median of its responses.
#[test]
fn absolute_error_predicts_the_leaf_median() {
    let features = array![[0.0], [1.0], [2.0]];
    let responses = array![1.0, 2.0, 6.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(0)
        .criterion(SKTreeCriterion::AbsoluteError)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(
        model.predict(features.view()).expect("predict"),
        array![2.0, 2.0, 2.0]
    );
}

/// A one-split squared-error tree predicts each side's mean.
#[test]
fn squared_error_split_predicts_child_means() {
    let features = array![[0.0], [0.0], [1.0], [1.0]];
    let responses = array![1.0, 2.0, 6.0, 10.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(1)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(model.node_count(), 3);
    assert_eq!(
        model.predict(features.view()).expect("predict"),
        array![1.5, 1.5, 8.0, 8.0]
    );
}

/// Sparse prediction matches dense prediction for the same logical design.
#[test]
fn sparse_prediction_matches_dense() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let responses = array![1.0, 2.0, 6.0, 10.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(1)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    let sparse = CsMat::new(
        (4, 1),
        vec![0, 0, 1, 2, 3],
        vec![0, 0, 0],
        vec![1.0, 2.0, 3.0],
    );
    assert_eq!(
        model.predict(features.view()).expect("dense"),
        model.predict(sparse.view()).expect("sparse")
    );
}

/// Integer targets are promoted to the model scalar.
#[test]
fn integer_targets_are_promoted() {
    let features = array![[0.0], [1.0]];
    let responses = array![2_i64, 4];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(
        model.predict(features.view()).expect("predict"),
        array![3.0, 3.0]
    );
}

/// Prediction rejects a feature count that differs from the fit.
#[test]
fn prediction_rejects_a_feature_mismatch() {
    let features = array![[0.0], [1.0]];
    let responses = array![1.0, 2.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert!(matches!(
        model.predict(array![[0.0, 1.0]].view()),
        Err(sciencekit_common::SKError::ShapeMismatch { .. })
    ));
}

/// The configured backend and execution intent reach the fitted model.
#[test]
fn backend_reaches_the_model() {
    let features = array![[0.0], [1.0]];
    let responses = array![1.0, 2.0];
    let model = SKDecisionTreeRegressor::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}

/// A strided continuous target is valid caller data: fitting must not panic.
#[test]
fn strided_continuous_targets_are_accepted() {
    use sciencekit_common::SKSupervisedFit;

    let features = ndarray::array![[0.0_f64], [1.0], [2.0], [3.0]];
    let wide_targets = ndarray::array![[0.0_f64, 9.0], [1.0, 9.0], [2.0, 9.0], [3.0, 9.0]];
    let strided = wide_targets.column(0);
    assert!(!strided.is_standard_layout());
    let model = SKDecisionTreeRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(&features, strided)
        .expect("a strided continuous target is valid");
    let predictions = model.predict(features.view()).expect("predict succeeds");
    assert_eq!(predictions.len(), 4);
}
