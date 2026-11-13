//! Mapping a sciencekit-exported preprocessing graph back to scaler state
//! (spec `native-model-import`, requirement "ONNX native import").
//!
//! The exporter writes the learned per-column statistics as named
//! initializers. Reading those names back is exact, while inferring
//! `(x - a) / b` versus `(x - a) * b` from operator order would be ambiguous
//! between the three scalers.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKFloat, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    sk_model_tensor_data_type,
};

use super::initializers::constant;
use super::metadata::SKOnnxNativeMetadata;
use crate::graph::SKOnnxModel;
use crate::native::error::SKOnnxNativeError;

/// Convert a scaler graph, or return `None` when the estimator is not a scaler.
pub(crate) fn state_from_scaler<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: &SKOnnxNativeMetadata,
) -> Result<Option<SKModelState<F>>, SKOnnxNativeError> {
    match metadata.estimator_name.as_str() {
        "SKStandardScaler" => standard_scaler(model, metadata).map(Some),
        "SKMinMaxScaler" => minmax_scaler(model, metadata).map(Some),
        "SKRobustScaler" => robust_scaler(model, metadata).map(Some),
        _ => Ok(None),
    }
}

fn standard_scaler<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: &SKOnnxNativeMetadata,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let features = metadata.feature_count;
    let mean =
        column(model, "standard_scaler.mean", features).unwrap_or_else(|| vec![0.0; features]);
    let scale =
        column(model, "standard_scaler.scale", features).unwrap_or_else(|| vec![1.0; features]);
    let sum_squared_deviations: Vec<F> = scale
        .iter()
        .map(|value| F::from(value * value).unwrap_or_else(F::zero))
        .collect();
    let metadata = scaler_metadata::<F>("SKStandardScaler", features)?
        .with_parameter("sample_count", SKParameterValue::from(1_i64))
        .map_err(native)?
        .with_parameter(
            "with_mean",
            SKParameterValue::from(model_has(model, "standard_scaler.mean")),
        )
        .map_err(native)?
        .with_parameter(
            "with_std",
            SKParameterValue::from(model_has(model, "standard_scaler.scale")),
        )
        .map_err(native)?;
    let tensors = vec![
        tensor("standard_scaler.mean", mean.clone())?,
        tensor("standard_scaler.scale", scale)?,
        tensor("standard_scaler.moments_mean", mean)?,
        SKTensor::floating(
            "standard_scaler.sum_squared_deviations",
            vec![features],
            sum_squared_deviations,
        )
        .map_err(native)?,
    ];
    SKModelState::new(metadata, tensors).map_err(native)
}

fn minmax_scaler<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: &SKOnnxNativeMetadata,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let features = metadata.feature_count;
    let minimum = required_column(model, "minmax_scaler.minimum", features)?;
    let scale = required_column(model, "minmax_scaler.scale", features)?;
    let feature_minimum = constant(model, "minmax_scaler.feature_minimum")
        .and_then(|constant| constant.scalar())
        .unwrap_or(0.0);
    let feature_maximum = constant(model, "minmax_scaler.feature_maximum")
        .and_then(|constant| constant.scalar())
        .unwrap_or(1.0);
    let range = feature_maximum - feature_minimum;
    let maximum: Vec<F> = minimum
        .iter()
        .zip(&scale)
        .map(|(minimum, scale)| F::from(minimum + scale * range).unwrap_or_else(F::zero))
        .collect();
    let data_range: Vec<F> = scale
        .iter()
        .map(|value| F::from(value * range).unwrap_or_else(F::zero))
        .collect();
    let metadata = scaler_metadata::<F>("SKMinMaxScaler", features)?
        .with_parameter("feature_minimum", SKParameterValue::from(feature_minimum))
        .map_err(native)?
        .with_parameter("feature_maximum", SKParameterValue::from(feature_maximum))
        .map_err(native)?
        .with_parameter(
            "clip",
            SKParameterValue::from(
                model
                    .model()
                    .graph
                    .nodes
                    .iter()
                    .any(|node| node.op_type == "Clip"),
            ),
        )
        .map_err(native)?;
    let tensors = vec![
        tensor("minmax_scaler.minimum", minimum)?,
        SKTensor::floating("minmax_scaler.maximum", vec![features], maximum).map_err(native)?,
        SKTensor::floating("minmax_scaler.data_range", vec![features], data_range)
            .map_err(native)?,
        SKTensor::floating(
            "minmax_scaler.scale",
            vec![features],
            scale
                .iter()
                .map(|value| F::from(*value).unwrap_or_else(F::zero))
                .collect(),
        )
        .map_err(native)?,
    ];
    SKModelState::new(metadata, tensors).map_err(native)
}

fn robust_scaler<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: &SKOnnxNativeMetadata,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let features = metadata.feature_count;
    let center =
        column(model, "robust_scaler.center", features).unwrap_or_else(|| vec![0.0; features]);
    let scale = required_column(model, "robust_scaler.scale", features)?;
    let metadata = scaler_metadata::<F>("SKRobustScaler", features)?;
    let tensors = vec![
        tensor("robust_scaler.center", center)?,
        tensor("robust_scaler.scale", scale)?,
    ];
    SKModelState::new(metadata, tensors).map_err(native)
}

fn scaler_metadata<F: SKFloat>(
    estimator_name: &str,
    features: usize,
) -> Result<SKModelMetadata, SKOnnxNativeError> {
    SKModelMetadata::new(
        estimator_name,
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        features,
    )
    .map_err(native)
}

fn tensor<F: SKFloat>(name: &str, values: Vec<f64>) -> Result<SKTensor<F>, SKOnnxNativeError> {
    SKTensor::floating(
        name,
        vec![values.len()],
        values
            .iter()
            .map(|value| F::from(*value).unwrap_or_else(F::zero))
            .collect(),
    )
    .map_err(native)
}

fn column(model: &SKOnnxModel, name: &str, features: usize) -> Option<Vec<f64>> {
    constant(model, name)
        .filter(|constant| constant.values.len() == features)
        .map(|constant| constant.values)
}

fn required_column(
    model: &SKOnnxModel,
    name: &str,
    features: usize,
) -> Result<Vec<f64>, SKOnnxNativeError> {
    column(model, name, features).ok_or_else(|| SKOnnxNativeError::InvalidGraph {
        detail: format!(
            "the scaler graph does not carry the initializer `{name}` of width {features}"
        ),
    })
}

fn model_has(model: &SKOnnxModel, name: &str) -> bool {
    constant(model, name).is_some()
}

fn native(error: sciencekit_common::SKError) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(sciencekit_interop::native::SKNativeImportError::from(error))
}
