//! Companion tests: continued fitting on an imported native model
//! (spec `native-model-import`, requirement "Continued training").

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit, SKTargetView};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;

use super::conversion::SKNativeImport;
use super::model::SKNativeClassifier;
use super::{sk_import_native_model_from_bytes, test_models};
use crate::safetensors::sk_export_model_safetensors;

fn extra_features() -> ndarray::Array2<f64> {
    array![[5.0_f64, 3.0], [6.0, 4.0]]
}

#[test]
fn native_retrain_logistic_regression_warm_starts_from_the_import() {
    let model = test_models::logistic_regression();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let mut imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    assert!(imported.supports_retraining());

    let extra_labels = ["low", "high"];
    let references = match &imported {
        SKNativeImport::Classifier {
            model: SKNativeClassifier::LogisticRegression(inner),
            ..
        } => SKLogisticRegression::<f64>::build()
            .warm_start_from(inner)
            .build()
            .unwrap()
            .fit(extra_features().view(), extra_labels.as_slice())
            .unwrap(),
        _ => panic!("the fixture must import as a logistic model"),
    };
    imported
        .partial_fit(
            extra_features().view(),
            SKTargetView::Nominal(extra_labels.as_slice()),
        )
        .unwrap();
    let features = test_models::features();
    assert_eq!(
        references.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
    assert_eq!(
        references.predict_probabilities(features.view()).unwrap(),
        imported.predict_probabilities(features.view()).unwrap()
    );
}

#[test]
fn native_retrain_rejects_an_inference_only_family() {
    let model = test_models::decision_tree_regressor();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let mut imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    assert!(!imported.supports_retraining());
    let extra = array![1.0_f64, 2.0];
    let error = imported
        .partial_fit(
            extra_features().view(),
            SKTargetView::Continuous(extra.view()),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        sciencekit_common::SKError::UnsupportedOperation {
            operation: "partial_fit",
            ..
        }
    ));
}

#[test]
fn native_retrain_rejects_an_ordinary_least_squares_state() {
    let model = test_models::linear_regression();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let mut imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    assert!(!imported.supports_retraining());
    let extra = array![1.0_f64, 2.0];
    assert!(
        imported
            .partial_fit(
                extra_features().view(),
                SKTargetView::Continuous(extra.view()),
            )
            .is_err()
    );
}

#[test]
fn native_retrain_sgd_regressor_accepts_more_data() {
    use sciencekit_linear_model::sgd_regressor::SKSGDRegressor;
    let targets = array![1.0_f64, 3.0, 4.0, 14.0, 11.0, 16.0];
    let model = SKSGDRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(test_models::features().view(), targets.view())
        .unwrap();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let mut imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    assert!(imported.supports_retraining());
    let extra = array![20.0_f64, 25.0];
    imported
        .partial_fit(
            extra_features().view(),
            SKTargetView::Continuous(extra.view()),
        )
        .unwrap();
    assert!(
        imported
            .predict_regression(test_models::features().view())
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn native_retrain_guidance_explains_why_a_family_is_inference_only() {
    let model = test_models::decision_tree_regressor();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let mut imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    let extra = array![1.0_f64, 2.0];
    let message = imported
        .partial_fit(
            extra_features().view(),
            SKTargetView::Continuous(extra.view()),
        )
        .unwrap_err()
        .to_string();
    assert!(message.contains("partial_fit"));
    assert!(message.contains("inference-only"));
}

#[test]
fn native_retrain_coverage_marks_inference_only_families() {
    for row in super::sk_native_import_coverage() {
        if !row.retrainable {
            assert!(
                !row.note.is_empty(),
                "{} must explain why it is inference-only",
                row.estimator_name
            );
        }
    }
}
