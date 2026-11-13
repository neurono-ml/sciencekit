//! Format-agnostic state export/import for [`SKMinMaxScalerModel`].

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_validate_import_metadata,
};

use super::SKMinMaxScalerModel;

/// Stable estimator identifier for exported min/max-scaler states.
pub const SK_MINMAX_SCALER_ESTIMATOR_NAME: &str = "SKMinMaxScaler";
/// The tensor carrying the per-column learned minimum.
pub const SK_MINMAX_SCALER_MINIMUM_TENSOR: &str = "minmax_scaler.minimum";
/// The tensor carrying the per-column learned maximum.
pub const SK_MINMAX_SCALER_MAXIMUM_TENSOR: &str = "minmax_scaler.maximum";
/// The tensor carrying the per-column learned range.
pub const SK_MINMAX_SCALER_RANGE_TENSOR: &str = "minmax_scaler.data_range";
/// The tensor carrying the per-column affine scale.
pub const SK_MINMAX_SCALER_SCALE_TENSOR: &str = "minmax_scaler.scale";

impl<F: SKFloat> SKExportable<F> for SKMinMaxScalerModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_MINMAX_SCALER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.minimum.len(),
        )?
        .with_parameter(
            "sample_count",
            SKParameterValue::from(self.sample_count as i64),
        )?
        .with_parameter(
            "feature_minimum",
            SKParameterValue::from(
                self.feature_minimum
                    .to_f64()
                    .expect("a finite feature range fits in f64"),
            ),
        )?
        .with_parameter(
            "feature_maximum",
            SKParameterValue::from(
                self.feature_maximum
                    .to_f64()
                    .expect("a finite feature range fits in f64"),
            ),
        )?
        .with_parameter("clip", SKParameterValue::from(self.clip))?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let columns = self.minimum.len();
        let tensors = vec![
            SKTensor::floating(
                SK_MINMAX_SCALER_MINIMUM_TENSOR,
                vec![columns],
                self.minimum.to_vec(),
            )?,
            SKTensor::floating(
                SK_MINMAX_SCALER_MAXIMUM_TENSOR,
                vec![columns],
                self.maximum.to_vec(),
            )?,
            SKTensor::floating(
                SK_MINMAX_SCALER_RANGE_TENSOR,
                vec![columns],
                self.data_range.to_vec(),
            )?,
            SKTensor::floating(
                SK_MINMAX_SCALER_SCALE_TENSOR,
                vec![columns],
                self.scale.to_vec(),
            )?,
        ];
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKMinMaxScalerModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_MINMAX_SCALER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let minimum = read_column(state, SK_MINMAX_SCALER_MINIMUM_TENSOR, feature_count)?;
        let maximum = read_column(state, SK_MINMAX_SCALER_MAXIMUM_TENSOR, feature_count)?;
        let data_range = read_column(state, SK_MINMAX_SCALER_RANGE_TENSOR, feature_count)?;
        let scale = read_column(state, SK_MINMAX_SCALER_SCALE_TENSOR, feature_count)?;
        let metadata = state.metadata();
        let feature_minimum = F::from(sk_float_parameter(metadata, "feature_minimum", 0.0)).ok_or(
            SKError::Conversion("feature minimum is not finite".to_string()),
        )?;
        let feature_maximum = F::from(sk_float_parameter(metadata, "feature_maximum", 1.0)).ok_or(
            SKError::Conversion("feature maximum is not finite".to_string()),
        )?;
        Ok(SKMinMaxScalerModel {
            sample_count: sk_integer_parameter(metadata, "sample_count", 1).max(0) as u64,
            minimum,
            maximum,
            data_range,
            scale,
            feature_minimum,
            feature_maximum,
            clip: sk_boolean_parameter(metadata, "clip", false),
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}

/// Read a one-dimensional tensor with exactly `feature_count` values.
fn read_column<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
    feature_count: usize,
) -> Result<Array1<F>, SKError> {
    let (_, values) = sk_read_floating_tensor(state, name)?;
    if values.len() != feature_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![feature_count],
            found: vec![values.len()],
        });
    }
    Ok(Array1::from_vec(values))
}
