//! Validation tests asserting the exact central error variants (task 7.1).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sprs::CsMat;

use crate::classifier::SKDecisionTreeClassifier;
use crate::regressor::SKDecisionTreeRegressor;
use crate::splitter::{SKMaximumFeatures, SKSplitter};

fn classifier() -> SKDecisionTreeClassifier<f64> {
    SKDecisionTreeClassifier::<f64>::build()
        .build()
        .expect("builds")
}

fn regressor() -> SKDecisionTreeRegressor<f64> {
    SKDecisionTreeRegressor::<f64>::build()
        .build()
        .expect("builds")
}

/// Zero rows are rejected before any element is processed.
#[test]
fn empty_input_is_rejected() {
    let features: Array2<f64> = Array2::zeros((0, 2));
    let labels = Array1::from_vec(vec![0_i64, 1]);
    let error = classifier()
        .fit(&features, labels.view())
        .expect_err("empty classifier fit must fail");
    assert!(matches!(error, SKError::EmptyInput { operation: "fit" }));
    let responses = Array1::from_vec(vec![0.0, 1.0]);
    let error = regressor()
        .fit(&features, responses.view())
        .expect_err("empty regressor fit must fail");
    assert!(matches!(error, SKError::EmptyInput { operation: "fit" }));
}

/// A non-finite feature is rejected at its flattened position during fit.
#[test]
fn non_finite_fit_values_are_rejected() {
    let features = array![[0.0, f64::NAN], [1.0, 2.0]];
    let labels = array![0_i64, 1];
    let error = classifier()
        .fit(&features, labels.view())
        .expect_err("NaN features must fail");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));

    let dense = array![[0.0, 1.0], [f64::INFINITY, 2.0]];
    let responses = array![0.0, 1.0];
    let error = regressor()
        .fit(&dense, responses.view())
        .expect_err("infinite features must fail");
    assert!(matches!(error, SKError::NonFiniteValue { position: 2 }));
}

/// A non-finite sparse entry is rejected without treating it as missing.
#[test]
fn non_finite_sparse_values_are_rejected() {
    let sparse = CsMat::new((2, 2), vec![0, 1, 2], vec![1, 0], vec![f64::NAN, 1.0]);
    let labels = array![0_i64, 1];
    let error = classifier()
        .fit(sparse.view(), labels.view())
        .expect_err("NaN sparse features must fail");
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}

/// Non-finite prediction values are rejected as well.
#[test]
fn non_finite_prediction_values_are_rejected() {
    let features = array![[0.0], [1.0]];
    let labels = array![0_i64, 1];
    let model = classifier()
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let error = model
        .predict_probabilities(array![[f64::NAN]].view())
        .expect_err("NaN prediction must fail");
    assert!(matches!(error, SKError::NonFiniteValue { position: 0 }));

    let responses = array![0.0, 1.0];
    let model = regressor()
        .fit(&features, responses.view())
        .expect("fit succeeds");
    let error = model
        .predict(array![[f64::INFINITY]].view())
        .expect_err("infinite prediction must fail");
    assert!(matches!(error, SKError::NonFiniteValue { position: 0 }));
}

/// A prediction with a different feature count is a shape mismatch.
#[test]
fn feature_count_mismatch_is_rejected() {
    let features = array![[0.0], [1.0]];
    let labels = array![0_i64, 1];
    let model = classifier()
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let error = model
        .predict_labels(array![[0.0, 1.0]].view())
        .expect_err("mismatched features must fail");
    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![1, 1]);
            assert_eq!(found, vec![1, 2]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

/// A single-class classification target is rejected as an invalid
/// hyperparameter naming the class requirement.
#[test]
fn single_class_classification_is_rejected() {
    let features = array![[0.0], [1.0], [2.0]];
    let labels = array![4_i64, 4, 4];
    let error = classifier()
        .fit(&features, labels.view())
        .expect_err("single-class fit must fail");
    match error {
        SKError::InvalidHyperparameter { name, reason } => {
            assert_eq!(name, "targets");
            assert!(reason.contains("two classes"));
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

/// A target vector whose length differs from the feature rows is rejected.
#[test]
fn target_length_mismatch_is_rejected() {
    let features = array![[0.0], [1.0], [2.0]];
    let labels = array![0_i64, 1];
    let error = classifier()
        .fit(&features, labels.view())
        .expect_err("short targets must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Continuous class targets and nominal regression targets are unsupported
/// representations.
#[test]
fn unsupported_target_representations_are_rejected() {
    let features = array![[0.0], [1.0]];
    let continuous = array![0.0, 1.0];
    let error = classifier()
        .fit(&features, continuous.view())
        .expect_err("continuous class labels must fail");
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
    let nominal = ["a", "b"];
    let error = regressor()
        .fit(&features, &nominal[..])
        .expect_err("nominal responses must fail");
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

/// Invalid depth/split/leaf and feature-subsampling configurations are
/// rejected at build time with the central error.
#[test]
fn invalid_hyperparameters_are_rejected() {
    let invalid =
        |builder: Result<SKDecisionTreeClassifier<f64>, SKError>, name: &str| match builder {
            Err(SKError::InvalidHyperparameter { name: actual, .. }) => assert_eq!(actual, name),
            _ => panic!("expected InvalidHyperparameter({name})"),
        };
    invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .minimum_samples_split(1)
            .build(),
        "minimum_samples_split",
    );
    invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .minimum_samples_leaf(0)
            .build(),
        "minimum_samples_leaf",
    );
    invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_features(SKMaximumFeatures::Count(0))
            .build(),
        "maximum_features",
    );
    invalid(
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_features(SKMaximumFeatures::Fraction(0.0))
            .build(),
        "maximum_features",
    );
    // A valid depth and splitter combination still builds.
    let _ = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(0)
        .splitter(SKSplitter::Random)
        .build()
        .expect("valid configuration builds");
}
