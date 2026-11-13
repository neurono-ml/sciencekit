//! Format-agnostic state export/import for [`SKRobustScalerModel`].
//!
//! The learned center and scale fully determine `transform`, so they are the
//! exported state. The per-column online sketches are an incremental-fitting
//! accelerator: they are not part of the learned transform and are rebuilt
//! empty on import, so an imported model transforms identically but cannot
//! resume `partial_fit` without new data.

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_validate_import_metadata,
};

use super::SKRobustScalerModel;
use crate::robust_scaler::configuration::SKRobustScalerConfiguration;

/// Stable estimator identifier for exported robust-scaler states.
pub const SK_ROBUST_SCALER_ESTIMATOR_NAME: &str = "SKRobustScaler";
/// The tensor carrying the per-column learned center.
pub const SK_ROBUST_SCALER_CENTER_TENSOR: &str = "robust_scaler.center";
/// The tensor carrying the per-column learned scale.
pub const SK_ROBUST_SCALER_SCALE_TENSOR: &str = "robust_scaler.scale";

impl<F: SKFloat> SKExportable<F> for SKRobustScalerModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_ROBUST_SCALER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.center.len(),
        )?
        .with_parameter(
            "sample_count",
            SKParameterValue::from(self.sample_count as i64),
        )?
        .with_parameter(
            "with_centering",
            SKParameterValue::from(self.configuration.with_centering),
        )?
        .with_parameter(
            "with_scaling",
            SKParameterValue::from(self.configuration.with_scaling),
        )?
        .with_parameter(
            "quantile_low",
            SKParameterValue::from(
                self.configuration
                    .quantile_low
                    .to_f64()
                    .expect("the quantile bounds fit in f64"),
            ),
        )?
        .with_parameter(
            "quantile_high",
            SKParameterValue::from(
                self.configuration
                    .quantile_high
                    .to_f64()
                    .expect("the quantile bounds fit in f64"),
            ),
        )?
        .with_parameter(
            "unit_variance",
            SKParameterValue::from(self.configuration.unit_variance),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let columns = self.center.len();
        let tensors = vec![
            SKTensor::floating(
                SK_ROBUST_SCALER_CENTER_TENSOR,
                vec![columns],
                self.center.to_vec(),
            )?,
            SKTensor::floating(
                SK_ROBUST_SCALER_SCALE_TENSOR,
                vec![columns],
                self.scale.to_vec(),
            )?,
        ];
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKRobustScalerModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_ROBUST_SCALER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let center = read_column(state, SK_ROBUST_SCALER_CENTER_TENSOR, feature_count)?;
        let scale = read_column(state, SK_ROBUST_SCALER_SCALE_TENSOR, feature_count)?;
        let metadata = state.metadata();
        let quantile_low = F::from(sk_float_parameter(metadata, "quantile_low", 25.0)).ok_or(
            SKError::Conversion("quantile_low is not finite".to_string()),
        )?;
        let quantile_high = F::from(sk_float_parameter(metadata, "quantile_high", 75.0)).ok_or(
            SKError::Conversion("quantile_high is not finite".to_string()),
        )?;
        let configuration = SKRobustScalerConfiguration {
            with_centering: sk_boolean_parameter(metadata, "with_centering", true),
            with_scaling: sk_boolean_parameter(metadata, "with_scaling", true),
            quantile_low,
            quantile_high,
            unit_variance: sk_boolean_parameter(metadata, "unit_variance", false),
        };
        Ok(SKRobustScalerModel {
            sample_count: sk_integer_parameter(metadata, "sample_count", 1).max(0) as u64,
            center,
            scale,
            sketches: Vec::new(),
            configuration,
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
