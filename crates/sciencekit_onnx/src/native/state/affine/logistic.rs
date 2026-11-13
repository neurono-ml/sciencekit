//! Mapping an affine graph with a link operator to a logistic state.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKFloat, SKModelMetadata, SKModelState, SKParameterValue,
    sk_model_tensor_data_type,
};

use super::decision::Affine;
use super::shared::{native, tensor};
use crate::native::error::SKOnnxNativeError;
use crate::native::state::metadata::SKOnnxNativeMetadata;

pub(crate) fn state<F: SKFloat>(
    metadata: Option<&SKOnnxNativeMetadata>,
    affine: Affine,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let feature_count = affine.feature_count;
    let heads = affine.weights.shape[1];
    let class_labels = metadata
        .and_then(|metadata| metadata.class_labels.clone())
        .unwrap_or_else(|| (0..heads.max(2)).map(|index| index.to_string()).collect());
    let class_count = class_labels.len();
    let coefficients = if heads == 1 {
        affine
            .weights
            .values
            .iter()
            .map(|value| F::from(*value).unwrap_or_else(F::zero))
            .collect::<Vec<F>>()
    } else {
        (0..heads)
            .flat_map(|head| {
                let values = &affine.weights.values;
                (0..feature_count).map(move |feature| values[feature * heads + head])
            })
            .map(|value| F::from(value).unwrap_or_else(F::zero))
            .collect()
    };
    let intercepts: Vec<F> = (0..heads)
        .map(|head| {
            affine
                .bias
                .as_ref()
                .and_then(|bias| bias.values.get(head).copied())
                .map(|value| F::from(value).unwrap_or_else(F::zero))
                .unwrap_or_else(F::zero)
        })
        .collect();
    let metadata = SKModelMetadata::new(
        "SKLogisticRegression",
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        feature_count,
    )
    .map_err(native)?
    .with_class_labels(class_labels)
    .map_err(native)?
    .with_parameter(
        "strategy",
        SKParameterValue::from(if heads == 1 { "binary" } else { "multinomial" }),
    )
    .map_err(native)?
    .with_parameter(
        "number_of_classes",
        SKParameterValue::from(class_count as i64),
    )
    .map_err(native)?
    .with_parameter(
        "include_intercept",
        SKParameterValue::from(affine.bias.is_some()),
    )
    .map_err(native)?;
    let tensors = vec![
        tensor(
            "logistic.coefficients",
            vec![heads, feature_count],
            coefficients,
        )?,
        tensor("logistic.intercepts", vec![heads], intercepts)?,
    ];
    SKModelState::new(metadata, tensors).map_err(native)
}
