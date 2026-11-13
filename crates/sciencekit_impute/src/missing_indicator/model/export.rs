//! Format-agnostic state export/import for [`SKMissingIndicatorModel`].

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_model_tensor_data_type, sk_parse_backend_parameter,
    sk_read_integer_tensor, sk_string_parameter, sk_validate_import_metadata,
};

use super::SKMissingIndicatorModel;
use crate::missing_indicator::SKMissingFeatureScope;

/// Stable estimator identifier for exported missing-indicator states.
pub const SK_MISSING_INDICATOR_ESTIMATOR_NAME: &str = "SKMissingIndicator";
/// The tensor carrying the feature columns the mask covers.
pub const SK_MISSING_INDICATOR_SCOPE_TENSOR: &str = "missing_indicator.scope";

impl<F: SKFloat> SKExportable<F> for SKMissingIndicatorModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_MISSING_INDICATOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_input_columns,
        )?
        .with_parameter(
            "features",
            SKParameterValue::from(match self.features {
                SKMissingFeatureScope::MissingOnly => "missing_only",
                SKMissingFeatureScope::All => "all",
            }),
        )?
        .with_parameter("error_on_new", SKParameterValue::from(self.error_on_new))?
        .with_parameter("sparse_output", SKParameterValue::from(self.sparse_output))?
        .with_parameter(
            "missing_values",
            missing_marker_parameter(self.missing_values),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let scope = SKTensor::integer(
            SK_MISSING_INDICATOR_SCOPE_TENSOR,
            vec![self.scope.len()],
            self.scope.iter().map(|&column| column as i64).collect(),
        )?;
        SKModelState::new(metadata, vec![scope])
    }
}

impl<F: SKFloat> SKImportable<F> for SKMissingIndicatorModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let number_of_input_columns = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_MISSING_INDICATOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            number_of_input_columns,
        )?;
        let columns = sk_read_integer_tensor(state, SK_MISSING_INDICATOR_SCOPE_TENSOR)?;
        if columns
            .iter()
            .any(|&column| column < 0 || column as usize >= number_of_input_columns)
        {
            return Err(SKError::ShapeMismatch {
                expected: vec![number_of_input_columns],
                found: vec![columns.len()],
            });
        }
        let scope: Vec<usize> = columns.iter().map(|&column| column as usize).collect();
        let metadata = state.metadata();
        let missing_values = read_missing_marker::<F>(metadata);
        Ok(SKMissingIndicatorModel {
            scope,
            number_of_input_columns,
            features: match sk_string_parameter(metadata, "features", "missing_only") {
                "all" => SKMissingFeatureScope::All,
                _ => SKMissingFeatureScope::MissingOnly,
            },
            error_on_new: sk_boolean_parameter(metadata, "error_on_new", true),
            sparse_output: sk_boolean_parameter(metadata, "sparse_output", false),
            missing_values,
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}

/// Encode a missing marker, spelling NaN as a string so two exports compare equal.
fn missing_marker_parameter<F: SKFloat>(marker: F) -> SKParameterValue {
    if marker.is_nan() {
        SKParameterValue::from("nan")
    } else {
        SKParameterValue::from(marker.to_f64().expect("a finite marker fits in f64"))
    }
}

/// Read a missing marker, restoring the NaN spelling.
fn read_missing_marker<F: SKFloat>(metadata: &SKModelMetadata) -> F {
    match metadata.parameter("missing_values") {
        Some(SKParameterValue::String(text)) if text == "nan" => F::nan(),
        Some(value) => value.as_float().and_then(F::from).unwrap_or_else(F::nan),
        None => F::nan(),
    }
}
