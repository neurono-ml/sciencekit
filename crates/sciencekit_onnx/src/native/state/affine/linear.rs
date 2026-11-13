//! Mapping an affine graph without a link to a linear-regression state.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKFloat, SKModelMetadata, SKModelState, SKParameterValue,
    sk_model_tensor_data_type,
};

use super::decision::Affine;
use super::shared::{native, tensor};
use crate::native::error::SKOnnxNativeError;

pub(crate) fn state<F: SKFloat>(affine: Affine) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let heads = affine.weights.shape[1];
    if heads != 1 {
        return Err(SKOnnxNativeError::UnsupportedOperator {
            operator: "MatMul".to_string(),
            reason: format!(
                "a regressor must emit one column, this weight emits {heads}; a multi-output affine graph is outside the subset"
            ),
        });
    }
    let feature_count = affine.feature_count;
    let coefficients = affine
        .weights
        .values
        .iter()
        .map(|value| F::from(*value).unwrap_or_else(F::zero))
        .collect::<Vec<F>>();
    let intercept = affine
        .bias
        .as_ref()
        .and_then(|bias| bias.scalar())
        .map(|value| F::from(value).unwrap_or_else(F::zero))
        .unwrap_or_else(F::zero);
    let metadata = SKModelMetadata::new(
        "SKLinearRegression",
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        feature_count,
    )
    .map_err(native)?
    .with_parameter(
        "include_intercept",
        SKParameterValue::from(affine.bias.is_some()),
    )
    .map_err(native)?;
    let tensors = vec![
        tensor("linear.coefficients", vec![feature_count], coefficients)?,
        tensor("linear.intercept", Vec::new(), vec![intercept])?,
        sciencekit_common::SKTensor::integer("linear.rank", Vec::new(), vec![0]).map_err(native)?,
        tensor("linear.singular_values", vec![0], Vec::new())?,
    ];
    SKModelState::new(metadata, tensors).map_err(native)
}
