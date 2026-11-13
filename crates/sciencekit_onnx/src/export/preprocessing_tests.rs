//! Preprocessing parity tests against the Rust transform and the reference
//! ONNX runtime.

use ndarray::Array2;
use sciencekit_common::{
    SKBuilder, SKExportable, SKFeatureTransformer, SKFloat, SKUnsupervisedFit,
};

use crate::acceptance_tests::{assert_regressor_interface, write_fixture};
use crate::graph::SKOnnxTensorDType;

use super::SKToOnnx;

fn matrix<F: SKFloat>() -> Array2<F> {
    Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 10.0, -2.0, 2.0, 20.0, 0.0, 3.0, 30.0, 2.0, 4.0, 40.0, 4.0, 5.0, 50.0, -4.0,
        ]
        .into_iter()
        .map(|value| F::from(value).expect("the fixture is finite"))
        .collect(),
    )
    .expect("the fixture matrix is well shaped")
}

fn json_matrix<F: SKFloat>(matrix: &Array2<F>) -> serde_json::Value {
    serde_json::Value::Array(
        matrix
            .rows()
            .into_iter()
            .map(|row| {
                serde_json::Value::Array(
                    row.iter()
                        .map(|value| {
                            serde_json::json!(value.to_f64().expect("the fixture is finite"))
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

fn json_matrix_from<F: SKFloat>(matrix: &ndarray::Array2<F>) -> serde_json::Value {
    json_matrix(matrix)
}

#[test]
fn standard_scaler_export_matches_the_rust_transform() {
    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

    let features: Array2<f64> = matrix();
    let estimator = SKStandardScaler::build()
        .with_mean(true)
        .with_std(true)
        .build()
        .expect("the scaler configuration is valid");
    let model = estimator.fit(&features).expect("the scaler fits");
    let transformed = model.transform(&features).expect("the scaler transforms");
    let onnx = model.to_onnx().expect("the scaler exports");
    assert_regressor_interface(&onnx, 3, 3, SKOnnxTensorDType::Double);
    assert_eq!(
        onnx.metadata()
            .get("sciencekit.estimator_name")
            .map(String::as_str),
        Some("SKStandardScaler")
    );
    write_fixture(
        "standard_scaler_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): json_matrix_from(&transformed) }),
        1e-9,
    );
}

#[test]
fn minmax_scaler_export_matches_the_clipped_transform() {
    use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;

    let features: Array2<f32> = matrix();
    let estimator = SKMinMaxScaler::<f32>::build()
        .feature_range((-1.0, 1.0))
        .clip(true)
        .build()
        .expect("the scaler configuration is valid");
    let model = estimator.fit(&features).expect("the scaler fits");
    let transformed = model.transform(&features).expect("the scaler transforms");
    let onnx = model.to_onnx().expect("the scaler exports");
    assert_regressor_interface(&onnx, 3, 3, SKOnnxTensorDType::Float);
    write_fixture(
        "minmax_scaler_f32_clipped",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): json_matrix_from(&transformed) }),
        1e-6,
    );
}

#[test]
fn robust_scaler_export_matches_the_rust_transform() {
    use sciencekit_preprocessing::robust_scaler::core_implementation::SKRobustScaler;

    let features: Array2<f64> = matrix();
    let estimator = SKRobustScaler::<f64>::build()
        .quantile_range((10.0, 90.0))
        .unit_variance(true)
        .build()
        .expect("the scaler configuration is valid");
    let model = estimator.fit(&features).expect("the scaler fits");
    let transformed = model.transform(&features).expect("the scaler transforms");
    let onnx = model.to_onnx().expect("the scaler exports");
    write_fixture(
        "robust_scaler_f64",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): json_matrix_from(&transformed) }),
        1e-9,
    );
}

#[test]
fn standard_scaler_without_mean_or_std_exports_an_identity_graph() {
    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

    let features: Array2<f32> = matrix();
    let estimator = SKStandardScaler::build()
        .with_mean(false)
        .with_std(false)
        .build()
        .expect("the scaler configuration is valid");
    let model = estimator.fit(&features).expect("the scaler fits");
    let onnx = model.to_onnx().expect("the scaler exports");
    assert_eq!(onnx.node_count(), 0);
    let state = model.export_state().expect("the scaler state exports");
    assert_eq!(
        state.metadata().estimator_name(),
        "SKStandardScaler",
        "the dispatcher matched the family by its exported name"
    );
}

#[test]
fn one_hot_encoder_exports_integer_codes_under_the_ignore_policy() {
    use sciencekit_preprocessing::one_hot_encoder::{SKOneHotEncoder, SKUnknownPolicy};

    let codes: Array2<i64> = ndarray::array![[3, 7], [4, 7], [3, 8]];
    let model = SKOneHotEncoder::build()
        .handle_unknown(SKUnknownPolicy::Ignore)
        .build()
        .expect("the encoder configuration is valid")
        .fit(&codes)
        .expect("the encoder fits");
    let encoded = model
        .transform_dense::<f64, _>(&codes)
        .expect("the encoder transforms");
    let onnx = SKToOnnx::<f64>::to_onnx(&model).expect("the encoder exports");
    assert_eq!(onnx.outputs()[0].feature_count(), Some(4));
    let features: Array2<f64> = codes.mapv(|code| code as f64);
    write_fixture(
        "one_hot_encoder_f64_integer",
        &onnx,
        "features",
        json_matrix(&features),
        serde_json::json!({ onnx.outputs()[0].name(): json_matrix(&encoded) }),
        1e-12,
    );
}

#[test]
fn one_hot_encoder_rejects_the_export_of_the_reject_policy() {
    use sciencekit_preprocessing::one_hot_encoder::SKOneHotEncoder;

    let codes: Array2<i64> = ndarray::array![[3, 7], [4, 7]];
    let model = SKOneHotEncoder::build()
        .build()
        .expect("the encoder configuration is valid")
        .fit(&codes)
        .expect("the encoder fits");
    let error = SKToOnnx::<f64>::to_onnx(&model).expect_err("the reject policy cannot be exported");
    assert!(matches!(
        error,
        sciencekit_common::SKError::UnsupportedOperation {
            operation: "onnx_export",
            ..
        }
    ));
}
