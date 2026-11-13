//! Metadata, dtype, size and concurrency acceptance tests (spec `onnx-export`,
//! requirement "Metadata" and "Validation").

use std::sync::Arc;

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKSupervisedFit, SKUnsupervisedFit};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::acceptance_tests::assert_regressor_interface;
use crate::graph::SKOnnxTensorDType;

use super::SKToOnnx;

fn scaler_for(
    features: &Array2<f64>,
) -> sciencekit_preprocessing::standard_scaler::model::SKStandardScalerModel<f64> {
    SKStandardScaler::build()
        .build()
        .expect("the scaler configuration is valid")
        .fit(features)
        .expect("the scaler fits")
}

#[test]
fn metadata_carries_producer_estimator_version_and_shape() {
    let features: Array2<f64> = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let model = scaler_for(&features);
    let onnx = model.to_onnx().expect("the scaler exports");
    let metadata = onnx.metadata();
    assert_eq!(metadata.get("sciencekit.estimator_name").map(String::as_str), Some("SKStandardScaler"));
    assert_eq!(metadata.get("sciencekit.format_version").map(String::as_str), Some("1"));
    assert_eq!(metadata.get("sciencekit.scalar_data_type").map(String::as_str), Some("F64"));
    assert_eq!(metadata.get("sciencekit.feature_count").map(String::as_str), Some("2"));
    assert_eq!(onnx.opset(), 17);
}

#[test]
fn dtype_mapping_declares_float_for_f32_and_double_for_f64() {
    let features: Array2<f32> = ndarray::array![[1.0], [2.0], [3.0]];
    let single = scaler_for(&features.mapv(|value| value as f64));
    let onnx = single.to_onnx().expect("the f64 scaler exports");
    assert_regressor_interface(&onnx, 1, 1, SKOnnxTensorDType::Double);

    use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler as Scaler;
    let single_precision = Scaler::build()
        .build()
        .expect("the scaler configuration is valid")
        .fit(&features)
        .expect("the scaler fits");
    let onnx = single_precision.to_onnx().expect("the f32 scaler exports");
    assert_regressor_interface(&onnx, 1, 1, SKOnnxTensorDType::Float);
}

#[test]
fn large_and_small_models_export_valid_graphs() {
    let wide = Array2::from_shape_fn((20, 400), |(row, column)| ((row + column) % 7) as f64);
    let model = scaler_for(&wide);
    let onnx = model.to_onnx().expect("the wide scaler exports");
    assert_regressor_interface(&onnx, 400, 400, SKOnnxTensorDType::Double);
    assert_eq!(onnx.initializer_count(), 2);

    let tiny: Array2<f64> = ndarray::array![[3.0]];
    let model = scaler_for(&tiny);
    let onnx = model.to_onnx().expect("the tiny scaler exports");
    assert_regressor_interface(&onnx, 1, 1, SKOnnxTensorDType::Double);
}

#[test]
fn concurrent_exports_of_one_model_produce_identical_bytes() {
    let features = Array2::from_shape_fn((10, 8), |(row, column)| (row * column) as f64 + 0.5);
    let model: Arc<sciencekit_preprocessing::standard_scaler::model::SKStandardScalerModel<f64>> =
        Arc::new(scaler_for(&features));
    let expected = model.to_onnx().expect("the scaler exports");
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = Arc::clone(&model);
            std::thread::spawn(move || model.to_onnx().expect("the concurrent export succeeds"))
        })
        .collect();
    for handle in handles {
        let model = handle.join().expect("the export thread joins");
        assert_eq!(model.bytes(), expected.bytes());
    }
}

#[test]
fn classifier_metadata_carries_the_class_table() {
    use sciencekit_linear_model::logistic_regression::SKLogisticRegression;

    let features: Array2<f64> = ndarray::array![
        [0.0, 0.0],
        [0.5, 0.2],
        [1.0, 1.0],
        [1.5, 1.2]
    ];
    let labels = vec![0_i64, 0, 1, 1];
    let model = SKLogisticRegression::build()
        .build()
        .expect("the classifier configuration is valid")
        .fit(features.view(), ndarray::Array1::from(labels).view())
        .expect("the classifier fits");
    let onnx = model.to_onnx().expect("the classifier exports");
    assert_eq!(
        onnx.metadata().get("sciencekit.number_of_classes").map(String::as_str),
        Some("2")
    );
    assert_eq!(
        onnx.metadata().get("sciencekit.class_label.0").map(String::as_str),
        Some("0")
    );
}
