//! Companion TDD tests for the linear SVC validation surface
//! (spec `linear-support-vector-classifier`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKError, SKSupervisedFit};
use sprs::CsMat;

use super::{SKLinearSVC, sk_linear_svc_loss_from_code};

#[test]
fn validation_rejects_empty_and_single_class_training_input() {
    let empty_features = Array2::<f64>::zeros((0, 3));
    let empty_labels = ndarray::Array1::<i64>::zeros(0);
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(empty_features.view(), empty_labels.view())
            .unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));

    let zero_columns = Array2::<f64>::zeros((4, 0));
    let labels = array![0_i64, 0, 1, 1];
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(zero_columns.view(), labels.view())
            .unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));

    let single_class_features = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let single_class_labels = array![7_i64, 7, 7];
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(single_class_features.view(), single_class_labels.view())
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "classes",
            ..
        }
    ));
}

#[test]
fn validation_rejects_target_length_mismatch_and_continuous_targets() {
    let features = array![[-1.0_f64, 0.0], [1.0, 0.0], [2.0, 1.0]];
    let short_labels = array![0_i64, 1];
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(features.view(), short_labels.view())
            .unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));

    let continuous_targets = array![-1.0_f64, 1.0, 2.0];
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(features.view(), continuous_targets.view())
            .unwrap_err(),
        SKError::UnsupportedRepresentation {
            representation: "continuous",
            ..
        }
    ));
}

#[test]
fn validation_rejects_non_finite_values_in_dense_and_sparse_inputs() {
    let nan_features = array![[-1.0_f64, 0.0], [f64::NAN, 0.0]];
    let labels = array![0_i64, 1];
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(nan_features.view(), labels.view())
            .unwrap_err(),
        SKError::NonFiniteValue { .. }
    ));

    let infinite_features = array![[-1.0_f64, 0.0], [f64::INFINITY, 0.0]];
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(infinite_features.view(), labels.view())
            .unwrap_err(),
        SKError::NonFiniteValue { .. }
    ));

    let sparse = CsMat::new((2, 2), vec![0, 1, 2], vec![0, 1], vec![-1.0_f64, f64::NAN]);
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .build()
            .unwrap()
            .fit(sparse.view(), labels.view())
            .unwrap_err(),
        SKError::NonFiniteValue { .. }
    ));
}

#[test]
fn validation_rejects_feature_count_mismatch_and_empty_queries() {
    let features = array![[-2.0_f64, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]];
    let labels = array![0_i64, 0, 1, 1];
    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let wrong_width = Array2::<f64>::zeros((3, 5));
    assert!(matches!(
        model.predict_labels(wrong_width.view()).unwrap_err(),
        SKError::ShapeMismatch {
            expected,
            found
        } if expected == vec![3, 2] && found == vec![3, 5]
    ));

    let empty_query = Array2::<f64>::zeros((0, 2));
    assert!(matches!(
        model.predict_labels(empty_query.view()).unwrap_err(),
        SKError::EmptyInput {
            operation: "predict"
        }
    ));
}

#[test]
fn validation_rejects_invalid_builder_hyperparameters_and_unsupported_loss() {
    assert!(matches!(
        SKLinearSVC::<f64>::build().C(0.0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build().C(-2.0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .tolerance(0.0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "tolerance",
            ..
        }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .maximum_number_of_iterations(0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "maximum_number_of_iterations",
            ..
        }
    ));
    assert!(matches!(
        sk_linear_svc_loss_from_code(3).unwrap_err(),
        SKError::InvalidHyperparameter { name: "loss", .. }
    ));
}
