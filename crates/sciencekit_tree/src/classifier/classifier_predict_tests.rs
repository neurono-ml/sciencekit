//! Classifier prediction tests (task 4.3).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sprs::CsMat;

use super::SKDecisionTreeClassifier;

fn fitted() -> super::SKDecisionTreeClassifierModel<f64> {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let labels = array![0_i64, 0, 1, 1];
    SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds")
}

/// A perfectly separable tree predicts the exact training labels.
#[test]
fn predict_labels_returns_the_majority_class() {
    let model = fitted();
    let features = array![[0.2], [1.2], [2.2], [3.8]];
    let labels = model.predict_labels(&features).expect("predict succeeds");
    assert_eq!(labels, array![0_i64, 0, 1, 1]);
}

/// Probabilities have one column per class and sum to one per row.
#[test]
fn predict_probabilities_sum_to_one() {
    let model = fitted();
    let features = array![[0.0], [2.0], [3.0]];
    let probabilities = model
        .predict_probabilities(&features)
        .expect("predict succeeds");
    assert_eq!(probabilities.shape(), &[3, 2]);
    for row in probabilities.rows() {
        let total: f64 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
    assert_eq!(probabilities.row(0).to_vec(), vec![1.0, 0.0]);
    assert_eq!(probabilities.row(2).to_vec(), vec![0.0, 1.0]);
}

/// A leaf with two class-0 and one class-1 samples predicts class 0 and stores
/// the normalized distribution.
#[test]
fn a_single_leaf_predicts_the_majority_class() {
    let features = array![[0.0], [1.0], [2.0]];
    let labels = array![0_i64, 0, 1];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let predicted = model.predict_labels(&features).expect("predict succeeds");
    assert_eq!(predicted, array![0_i64, 0, 0]);
    let probabilities = model
        .predict_probabilities(&features)
        .expect("predict succeeds");
    let expected = [2.0 / 3.0, 1.0 / 3.0];
    for (actual, expected) in probabilities.row(0).iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

/// Sparse prediction reads the stored entries and matches the dense result.
#[test]
fn sparse_prediction_matches_dense() {
    let model = fitted();
    let dense = array![[0.0], [2.0], [3.0]];
    let sparse = CsMat::new((3, 1), vec![0, 0, 1, 2], vec![0, 0], vec![2.0, 3.0]);
    let dense_probabilities = model.predict_probabilities(&dense).expect("dense");
    let sparse_probabilities = model.predict_probabilities(sparse.view()).expect("sparse");
    assert_eq!(dense_probabilities, sparse_probabilities);
    assert_eq!(
        model.predict_labels(sparse.view()).expect("sparse"),
        model.predict_labels(&dense).expect("dense")
    );
}

/// Prediction rejects a feature count that differs from the fit.
#[test]
fn prediction_rejects_a_feature_mismatch() {
    let model = fitted();
    let features = array![[0.0, 1.0]];
    let error = model
        .predict_probabilities(&features)
        .expect_err("mismatched features must fail");
    assert!(matches!(
        error,
        sciencekit_common::SKError::ShapeMismatch { .. }
    ));
}

/// The class table stays in canonical order and the probability columns follow
/// it.
#[test]
fn probability_columns_follow_the_class_table() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    let labels = array![5_i64, 5, 9, 9];
    let model = SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert_eq!(model.classes().label_of(0), Some("5"));
    assert_eq!(model.classes().label_of(1), Some("9"));
    let probabilities = model
        .predict_probabilities(array![[0.0]].view())
        .expect("predict succeeds");
    assert_eq!(probabilities.row(0).to_vec(), vec![1.0, 0.0]);
}
