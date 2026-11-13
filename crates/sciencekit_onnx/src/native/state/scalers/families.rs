//! The three scaler families the native ONNX mapper rebuilds.

use sciencekit_common::{SKFloat, SKModelState, SKParameterValue, SKTensor};

use super::state::{column, model_has, native, required_column, scalar, scaler_metadata, tensor};
use crate::graph::SKOnnxModel;
use crate::native::error::SKOnnxNativeError;
use crate::native::state::metadata::SKOnnxNativeMetadata;

pub(super) fn standard_scaler<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: &SKOnnxNativeMetadata,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let features = metadata.feature_count;
    let mean =
        column(model, "standard_scaler.mean", features)?.unwrap_or_else(|| vec![0.0; features]);
    let scale =
        column(model, "standard_scaler.scale", features)?.unwrap_or_else(|| vec![1.0; features]);
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

pub(super) fn minmax_scaler<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: &SKOnnxNativeMetadata,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let features = metadata.feature_count;
    let minimum = required_column(model, "minmax_scaler.minimum", features)?;
    let scale = required_column(model, "minmax_scaler.scale", features)?;
    let feature_minimum = scalar(model, "minmax_scaler.feature_minimum")?.unwrap_or(0.0);
    let feature_maximum = scalar(model, "minmax_scaler.feature_maximum")?.unwrap_or(1.0);
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

pub(super) fn robust_scaler<F: SKFloat>(
    model: &SKOnnxModel,
    metadata: &SKOnnxNativeMetadata,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let features = metadata.feature_count;
    let center =
        column(model, "robust_scaler.center", features)?.unwrap_or_else(|| vec![0.0; features]);
    let scale = required_column(model, "robust_scaler.scale", features)?;
    let metadata = scaler_metadata::<F>("SKRobustScaler", features)?;
    let tensors = vec![
        tensor("robust_scaler.center", center)?,
        tensor("robust_scaler.scale", scale)?,
    ];
    SKModelState::new(metadata, tensors).map_err(native)
}
