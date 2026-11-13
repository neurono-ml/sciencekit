//! Mapping a sciencekit-exported preprocessing graph back to scaler state
//! (spec `native-model-import`, requirement "ONNX native import").
//!
//! The exporter writes the learned per-column statistics as named
//! initializers. Reading those names back is exact, while inferring
//! `(x - a) / b` versus `(x - a) * b` from operator order would be ambiguous
//! between the three scalers.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKFloat, SKModelMetadata, SKModelState, SKTensor,
    sk_model_tensor_data_type,
};

use super::super::initializers::constant;
use super::super::metadata::SKOnnxNativeMetadata;
use super::super::validation::sk_require_floating;
use super::families::{minmax_scaler, robust_scaler, standard_scaler};
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

pub(super) fn scaler_metadata<F: SKFloat>(
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

pub(super) fn tensor<F: SKFloat>(
    name: &str,
    values: Vec<f64>,
) -> Result<SKTensor<F>, SKOnnxNativeError> {
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

pub(super) fn column(
    model: &SKOnnxModel,
    name: &str,
    features: usize,
) -> Result<Option<Vec<f64>>, SKOnnxNativeError> {
    let Some(constant) = constant(model, name) else {
        return Ok(None);
    };
    sk_require_floating(&constant, name)?;
    Ok((constant.values.len() == features).then_some(constant.values))
}

pub(super) fn required_column(
    model: &SKOnnxModel,
    name: &str,
    features: usize,
) -> Result<Vec<f64>, SKOnnxNativeError> {
    column(model, name, features)?.ok_or_else(|| SKOnnxNativeError::InvalidGraph {
        detail: format!(
            "the scaler graph does not carry the initializer `{name}` of width {features}"
        ),
    })
}

pub(super) fn scalar(model: &SKOnnxModel, name: &str) -> Result<Option<f64>, SKOnnxNativeError> {
    let Some(constant) = constant(model, name) else {
        return Ok(None);
    };
    sk_require_floating(&constant, name)?;
    Ok(constant.scalar())
}

pub(super) fn model_has(model: &SKOnnxModel, name: &str) -> bool {
    constant(model, name).is_some()
}

pub(super) fn native(error: sciencekit_common::SKError) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(sciencekit_interop::native::SKNativeImportError::from(error))
}
