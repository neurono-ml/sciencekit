//! Coverage and unsupported-family tests (spec `onnx-export`, requirement
//! "Per-family operators" and scenario "Unsupported model is explicit").

use sciencekit_common::{
    SKError, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType, SKTensorValues,
};

use crate::acceptance_tests::assert_regressor_interface;
use crate::graph::SKOnnxTensorDType;

use super::{sk_export_state_to_onnx, sk_onnx_coverage, sk_onnx_supported_families};

#[test]
fn export_coverage_lists_every_implemented_estimator() {
    let supported = sk_onnx_supported_families();
    for estimator in [
        "SKStandardScaler",
        "SKMinMaxScaler",
        "SKRobustScaler",
        "SKOneHotEncoder",
        "SKLinearRegression",
        "SKRidge",
        "SKLasso",
        "SKElasticNet",
        "SKSGDRegressor",
        "SKLogisticRegression",
        "SKDecisionTreeClassifier",
        "SKDecisionTreeRegressor",
        "SKRandomForestClassifier",
        "SKRandomForestRegressor",
        "SKKNeighborsClassifier",
        "SKKNeighborsRegressor",
        "SKKMeans",
        "SKMiniBatchKMeans",
        "SKPCA",
        "SKTruncatedSVD",
        "SKPipelineClassifier",
        "SKPipelineRegressor",
        "SKSVC",
        "SKSVR",
        "SKLinearSVC",
    ] {
        assert!(
            supported.contains(&estimator),
            "the coverage matrix is missing `{estimator}`"
        );
    }
}

#[test]
fn export_coverage_marks_unsupported_families_explicitly() {
    for row in sk_onnx_coverage().iter().filter(|row| !row.supported) {
        assert!(
            row.note.contains("UnsupportedOperation") || !row.note.is_empty(),
            "the unsupported family `{}` carries no reason",
            row.family
        );
    }
}

#[test]
fn export_coverage_rejects_an_unregistered_family_with_its_name() {
    let state: SKModelState<f32> = sk_state_named("SKGradientBoostingClassifier", 2);
    let error = sk_export_state_to_onnx(&state).expect_err("gradient boosting has no exporter");
    let reason = match error {
        SKError::UnsupportedOperation {
            operation: "onnx_export",
            reason,
        } => reason,
        other => panic!("unexpected error: {other}"),
    };
    assert!(reason.contains("SKGradientBoostingClassifier"));
    assert!(reason.contains("sk_onnx_coverage"));
}

#[test]
fn export_coverage_rejects_an_empty_state() {
    let metadata = SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F32, 2)
        .expect("the metadata is valid");
    let state: SKModelState<f32> =
        SKModelState::new(metadata, Vec::new()).expect("the state is valid");
    let error = sk_export_state_to_onnx(&state).expect_err("an empty state is rejected");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "onnx_model_state",
            ..
        }
    ));
}

#[test]
fn export_coverage_accepts_a_one_feature_model() {
    let metadata = SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F32, 1)
        .expect("the metadata is valid");
    let values = SKTensor::new(
        "standard_scaler.mean",
        SKTensorDataType::F32,
        vec![1],
        SKTensorValues::Floating(vec![0.0]),
    )
    .expect("the tensor is valid");
    let scale = SKTensor::new(
        "standard_scaler.scale",
        SKTensorDataType::F32,
        vec![1],
        SKTensorValues::Floating(vec![1.0]),
    )
    .expect("the tensor is valid");
    let state: SKModelState<f32> =
        SKModelState::new(metadata, vec![values, scale]).expect("the state is valid");
    let model = sk_export_state_to_onnx(&state).expect("the one-feature scaler exports");
    assert_regressor_interface(&model, 1, 1, SKOnnxTensorDType::Float);
}

fn sk_state_named(estimator_name: &str, feature_count: usize) -> SKModelState<f32> {
    let metadata = SKModelMetadata::new(estimator_name, 1, SKTensorDataType::F32, feature_count)
        .expect("the metadata is valid");
    let tensor = SKTensor::new(
        "placeholder.values",
        SKTensorDataType::F32,
        vec![feature_count],
        SKTensorValues::Floating(vec![0.0; feature_count]),
    )
    .expect("the tensor is valid");
    SKModelState::new(metadata, vec![tensor]).expect("the state is valid")
}
