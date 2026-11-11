//! Companion validation tests asserting exact central `SKError` variants
//! (task 4.1).

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKExecutionMode, SKRegressorPredictor,
    SKSupervisedFit,
};
use sciencekit_tree::SKClassWeight;
use sprs::CsMat;

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

fn default_classifier() -> SKRandomForestClassifier<f64> {
    SKRandomForestClassifier::<f64>::build()
        .build()
        .expect("builds")
}

fn default_regressor() -> SKRandomForestRegressor<f64> {
    SKRandomForestRegressor::<f64>::build()
        .build()
        .expect("builds")
}

/// Zero-row matrices are rejected with `EmptyInput` naming `"fit"` or `"predict"`.
#[test]
fn empty_input_is_rejected_on_fit_and_predict() {
    let empty_dense: Array2<f64> = Array2::zeros((0, 2));
    let empty_sparse: CsMat<f64> = CsMat::empty(sprs::CSR, 2);
    let (labels, responses) = (array![0_i64, 1], array![1.0, 2.0]);

    assert!(matches!(
        default_classifier().fit(&empty_dense, labels.view()),
        Err(SKError::EmptyInput { operation: "fit" })
    ));
    assert!(matches!(
        default_classifier().fit(empty_sparse.view(), labels.view()),
        Err(SKError::EmptyInput { operation: "fit" })
    ));
    assert!(matches!(
        default_regressor().fit(&empty_dense, responses.view()),
        Err(SKError::EmptyInput { operation: "fit" })
    ));

    let train = array![[0.0, 1.0], [2.0, 3.0]];
    let cls_model = default_classifier()
        .fit(&train, labels.view())
        .expect("fit");
    let reg_model = default_regressor()
        .fit(&train, responses.view())
        .expect("fit");
    assert!(matches!(
        cls_model.predict_labels(&empty_dense),
        Err(SKError::EmptyInput {
            operation: "predict"
        })
    ));
    assert!(matches!(
        reg_model.predict(&empty_dense),
        Err(SKError::EmptyInput {
            operation: "predict"
        })
    ));
}

/// Zero columns, wrong prediction feature counts, target length mismatches and
/// explicit class-weight mismatches return `ShapeMismatch`.
#[test]
fn shape_and_feature_count_mismatches_are_rejected() {
    let zero_columns: Array2<f64> = Array2::zeros((2, 0));
    let labels = array![0_i64, 1];
    assert!(matches!(
        default_classifier().fit(&zero_columns, labels.view()),
        Err(SKError::ShapeMismatch { .. })
    ));

    let train = array![[0.0, 1.0], [2.0, 3.0]];
    let cls_model = default_classifier()
        .fit(&train, labels.view())
        .expect("fit");
    let reg_model = default_regressor()
        .fit(&train, array![1.0, 2.0].view())
        .expect("fit");
    match cls_model.predict_probabilities(array![[1.0, 2.0, 3.0]].view()) {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!((expected, found), (vec![1, 2], vec![1, 3]));
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
    match reg_model.predict(array![[1.0]].view()) {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!((expected, found), (vec![1, 2], vec![1, 1]));
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }

    let three_rows = array![[0.0], [1.0], [2.0]];
    assert!(matches!(
        default_classifier().fit(&three_rows, array![0_i64, 1].view()),
        Err(SKError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        default_regressor().fit(&three_rows, array![1.0, 2.0].view()),
        Err(SKError::ShapeMismatch { .. })
    ));

    let wrong_weights = SKRandomForestClassifier::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 2.0, 3.0]))
        .build()
        .expect("builds");
    assert!(matches!(
        wrong_weights.fit(&three_rows, array![0_i64, 0, 1].view()),
        Err(SKError::ShapeMismatch { .. })
    ));
}

/// Non-finite values in dense features, sparse features, regression responses
/// and prediction batches are rejected at their exact positions.
#[test]
fn non_finite_values_are_rejected_at_exact_positions() {
    let labels = array![0_i64, 1];
    let responses = array![1.0, 2.0];
    let bad_sparse = CsMat::new((2, 2), vec![0, 1, 2], vec![1, 0], vec![f64::NAN, 1.0]);

    assert!(matches!(
        default_classifier().fit(&array![[0.0, f64::NAN], [1.0, 2.0]], labels.view()),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
    assert!(matches!(
        default_regressor().fit(&array![[0.0, 1.0], [f64::INFINITY, 2.0]], responses.view()),
        Err(SKError::NonFiniteValue { position: 2 })
    ));
    assert!(matches!(
        default_classifier().fit(bad_sparse.view(), labels.view()),
        Err(SKError::NonFiniteValue { position: 1 })
    ));

    let clean = array![[0.0, 1.0], [2.0, 3.0]];
    assert!(matches!(
        default_regressor().fit(&clean, array![1.0, f64::NEG_INFINITY].view()),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
    let cls_model = default_classifier()
        .fit(&clean, labels.view())
        .expect("fit");
    let reg_model = default_regressor()
        .fit(&clean, responses.view())
        .expect("fit");
    assert!(matches!(
        cls_model.predict_labels(array![[0.0, f64::NAN]].view()),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
    assert!(matches!(
        reg_model.predict(bad_sparse.view()),
        Err(SKError::NonFiniteValue { position: 1 })
    ));
}

/// Single-class classification targets return `InvalidHyperparameter` naming
/// `"targets"` and the two-class requirement.
#[test]
fn single_class_classification_is_rejected() {
    let features = array![[0.0], [1.0], [2.0]];
    match default_classifier().fit(&features, array![9_i64, 9, 9].view()) {
        Err(SKError::InvalidHyperparameter { name, reason }) => {
            assert_eq!(name, "targets");
            assert!(reason.contains("two classes"));
        }
        other => panic!("expected InvalidHyperparameter(targets), found {other:?}"),
    }
}

/// Wrong target families and incompatible streaming execution modes fail with
/// exact central variants.
#[test]
fn target_representations_and_incompatible_execution_modes_are_rejected() {
    let features = array![[0.0], [1.0], [2.0], [3.0]];
    assert!(matches!(
        default_classifier().fit(&features, array![0.0, 1.0, 2.0, 3.0].view()),
        Err(SKError::UnsupportedRepresentation { .. })
    ));
    assert!(matches!(
        default_regressor().fit(&features, &["a", "b", "a", "b"][..]),
        Err(SKError::UnsupportedRepresentation { .. })
    ));

    let streaming_cls = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(4)
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .expect("builds");
    assert!(matches!(
        streaming_cls.fit(&features, array![0_i64, 0, 1, 1].view()),
        Err(SKError::ExecutionModeIncompatible {
            mode: "out-of-core-streaming",
            pattern: "random-access",
        })
    ));
}
