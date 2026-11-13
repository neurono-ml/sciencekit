//! Format-agnostic state export/import for [`SKStandardScalerModel`].

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_model_tensor_data_type, sk_parse_backend_parameter, sk_read_floating_tensor,
    sk_validate_import_metadata,
};

use super::SKStandardScalerModel;

/// Stable estimator identifier for exported standard-scaler states.
pub const SK_STANDARD_SCALER_ESTIMATOR_NAME: &str = "SKStandardScaler";
/// The tensor carrying the per-column masked mean.
pub const SK_STANDARD_SCALER_MEAN_TENSOR: &str = "standard_scaler.mean";
/// The tensor carrying the per-column derived scale.
pub const SK_STANDARD_SCALER_SCALE_TENSOR: &str = "standard_scaler.scale";
/// The tensor carrying the running raw column mean.
pub const SK_STANDARD_SCALER_MOMENTS_MEAN_TENSOR: &str = "standard_scaler.moments_mean";
/// The tensor carrying the running sum of squared deviations.
pub const SK_STANDARD_SCALER_DEVIATIONS_TENSOR: &str = "standard_scaler.sum_squared_deviations";

impl<F: SKFloat> SKExportable<F> for SKStandardScalerModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_STANDARD_SCALER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.mean.len(),
        )?
        .with_parameter(
            "sample_count",
            SKParameterValue::from(self.sample_count as i64),
        )?
        .with_parameter("with_mean", SKParameterValue::from(self.with_mean))?
        .with_parameter("with_std", SKParameterValue::from(self.with_std))?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let columns = self.mean.len();
        let tensors = vec![
            SKTensor::floating(
                SK_STANDARD_SCALER_MEAN_TENSOR,
                vec![columns],
                self.mean.to_vec(),
            )?,
            SKTensor::floating(
                SK_STANDARD_SCALER_SCALE_TENSOR,
                vec![columns],
                self.scale.to_vec(),
            )?,
            SKTensor::floating(
                SK_STANDARD_SCALER_MOMENTS_MEAN_TENSOR,
                vec![columns],
                self.moments_mean.to_vec(),
            )?,
            SKTensor::floating(
                SK_STANDARD_SCALER_DEVIATIONS_TENSOR,
                vec![columns],
                self.sum_squared_deviations.to_vec(),
            )?,
        ];
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKStandardScalerModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_STANDARD_SCALER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let mean = read_column(state, SK_STANDARD_SCALER_MEAN_TENSOR, feature_count)?;
        let scale = read_column(state, SK_STANDARD_SCALER_SCALE_TENSOR, feature_count)?;
        let moments_mean =
            read_column(state, SK_STANDARD_SCALER_MOMENTS_MEAN_TENSOR, feature_count)?;
        let sum_squared_deviations =
            read_column(state, SK_STANDARD_SCALER_DEVIATIONS_TENSOR, feature_count)?;
        let metadata = state.metadata();
        Ok(SKStandardScalerModel {
            sample_count: sciencekit_common::sk_integer_parameter(metadata, "sample_count", 1)
                .max(0) as u64,
            moments_mean,
            sum_squared_deviations,
            mean,
            scale,
            with_mean: sciencekit_common::sk_boolean_parameter(metadata, "with_mean", true),
            with_std: sciencekit_common::sk_boolean_parameter(metadata, "with_std", true),
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
