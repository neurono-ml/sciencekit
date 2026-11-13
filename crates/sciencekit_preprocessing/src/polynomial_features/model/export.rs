//! Format-agnostic state export/import for [`SKPolynomialFeaturesModel`].

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_model_tensor_data_type, sk_parse_backend_parameter,
    sk_read_integer_tensor, sk_validate_import_metadata,
};

use super::SKPolynomialFeaturesModel;

/// Stable estimator identifier for exported polynomial-feature states.
pub const SK_POLYNOMIAL_FEATURES_ESTIMATOR_NAME: &str = "SKPolynomialFeatures";
/// The tensor carrying the flattened exponent terms.
pub const SK_POLYNOMIAL_TERMS_TENSOR: &str = "polynomial_features.terms";
/// The tensor carrying each term's start offset plus the final length.
pub const SK_POLYNOMIAL_TERM_OFFSETS_TENSOR: &str = "polynomial_features.term_offsets";

impl<F: SKFloat> SKExportable<F> for SKPolynomialFeaturesModel {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_POLYNOMIAL_FEATURES_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_input_features,
        )?
        .with_parameter(
            "number_of_output_features",
            SKParameterValue::from(self.terms.len() as i64),
        )?
        .with_parameter("sparse_output", SKParameterValue::from(self.sparse_output))?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut flattened: Vec<i64> = Vec::new();
        let mut offsets = Vec::with_capacity(self.terms.len() + 1);
        offsets.push(0_i64);
        for term in &self.terms {
            flattened.extend(term.iter().map(|&exponent| exponent as i64));
            offsets.push(flattened.len() as i64);
        }
        let terms =
            SKTensor::integer(SK_POLYNOMIAL_TERMS_TENSOR, vec![flattened.len()], flattened)?;
        let offsets = SKTensor::integer(
            SK_POLYNOMIAL_TERM_OFFSETS_TENSOR,
            vec![offsets.len()],
            offsets,
        )?;
        SKModelState::new(metadata, vec![terms, offsets])
    }
}

impl<F: SKFloat> SKImportable<F> for SKPolynomialFeaturesModel {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_POLYNOMIAL_FEATURES_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let flattened = sk_read_integer_tensor(state, SK_POLYNOMIAL_TERMS_TENSOR)?;
        let offsets = sk_read_integer_tensor(state, SK_POLYNOMIAL_TERM_OFFSETS_TENSOR)?;
        if offsets.len() < 2 {
            return Err(SKError::ShapeMismatch {
                expected: vec![2],
                found: vec![offsets.len()],
            });
        }
        let mut terms: Vec<Vec<usize>> = Vec::with_capacity(offsets.len() - 1);
        for window in offsets.windows(2) {
            let start = usize::try_from(window[0]).map_err(|_| SKError::ShapeMismatch {
                expected: vec![0],
                found: vec![window[0] as usize],
            })?;
            let end = usize::try_from(window[1]).map_err(|_| SKError::ShapeMismatch {
                expected: vec![0],
                found: vec![window[1] as usize],
            })?;
            let exponents = flattened.get(start..end).ok_or(SKError::ShapeMismatch {
                expected: vec![flattened.len()],
                found: vec![end],
            })?;
            terms.push(
                exponents
                    .iter()
                    .map(|&exponent| exponent.max(0) as usize)
                    .collect(),
            );
        }
        if terms.iter().flatten().any(|&index| index >= feature_count) {
            return Err(SKError::ShapeMismatch {
                expected: vec![feature_count],
                found: vec![terms.iter().flatten().copied().max().unwrap_or(0) + 1],
            });
        }
        let metadata = state.metadata();
        Ok(SKPolynomialFeaturesModel {
            terms,
            number_of_input_features: feature_count,
            sparse_output: sk_boolean_parameter(metadata, "sparse_output", false),
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}
