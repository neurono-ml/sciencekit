//! Format-agnostic state export/import for [`SKSVRModel`].

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKBuilderState, SKError, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_integer_parameter, sk_model_tensor_data_type, sk_parse_backend_parameter,
    sk_read_floating_tensor, sk_read_integer_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::SKSVRModel;
use crate::export_support::{
    SK_SUPPORT_VECTOR_REPRESENTATION, export_support_vectors, import_support_vectors,
    read_kernel_parameters, with_kernel_parameters,
};

/// Stable estimator identifier for exported SVR states.
pub const SK_SVR_ESTIMATOR_NAME: &str = "SKSVR";
/// The tensor carrying the signed dual coefficients.
pub const SK_SVR_DUAL_COEFFICIENTS_TENSOR: &str = "svr.dual_coefficients";
/// The tensor carrying the recovered bias.
pub const SK_SVR_BIAS_TENSOR: &str = "svr.bias";
/// The tensor carrying the original support-vector row indices.
pub const SK_SVR_SUPPORT_INDICES_TENSOR: &str = "svr.support_training_indices";

impl<F: SKFloat> SKExportable<F> for SKSVRModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = with_kernel_parameters(
            SKModelMetadata::new(
                SK_SVR_ESTIMATOR_NAME,
                SK_MODEL_FORMAT_VERSION,
                sk_model_tensor_data_type::<F>(),
                self.number_of_features,
            )?
            .with_parameter("iterations", SKParameterValue::from(self.iterations as i64))?
            .with_parameter("backend", sk_backend_parameter(self.backend))?,
            &self.kernel_parameters,
        )?;
        let (mut tensors, representation) = export_support_vectors(&self.support_vectors, "svr")?;
        metadata = metadata.with_parameter(
            SK_SUPPORT_VECTOR_REPRESENTATION,
            SKParameterValue::from(representation),
        )?;
        tensors.push(SKTensor::floating(
            SK_SVR_DUAL_COEFFICIENTS_TENSOR,
            vec![self.dual_coefficients.len()],
            self.dual_coefficients.to_vec(),
        )?);
        tensors.push(SKTensor::floating(
            SK_SVR_BIAS_TENSOR,
            Vec::new(),
            vec![self.bias],
        )?);
        tensors.push(SKTensor::integer(
            SK_SVR_SUPPORT_INDICES_TENSOR,
            vec![self.support_training_indices.len()],
            self.support_training_indices
                .iter()
                .map(|&index| index as i64)
                .collect(),
        )?);
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKSVRModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_SVR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let representation =
            sk_string_parameter(state.metadata(), SK_SUPPORT_VECTOR_REPRESENTATION, "dense");
        let support_vectors = import_support_vectors(state, "svr", representation)?;
        let (_, dual_coefficients) =
            sk_read_floating_tensor(state, SK_SVR_DUAL_COEFFICIENTS_TENSOR)?;
        let (_, bias_values) = sk_read_floating_tensor(state, SK_SVR_BIAS_TENSOR)?;
        let &[bias] = bias_values.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1],
                found: vec![bias_values.len()],
            });
        };
        let support_training_indices: Vec<usize> =
            sk_read_integer_tensor(state, SK_SVR_SUPPORT_INDICES_TENSOR)?
                .into_iter()
                .map(|index| index.max(0) as usize)
                .collect();
        if support_training_indices.len() != dual_coefficients.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![dual_coefficients.len()],
                found: vec![support_training_indices.len()],
            });
        }
        let metadata = state.metadata();
        Ok(SKSVRModel {
            support_vectors,
            support_training_indices,
            dual_coefficients: Array1::from_vec(dual_coefficients),
            bias,
            iterations: sk_integer_parameter(metadata, "iterations", 0).max(0) as usize,
            kernel_parameters: read_kernel_parameters(metadata)?,
            number_of_features: feature_count,
            backend: sk_parse_backend_parameter(metadata),
            builder_state: SKBuilderState::new(),
        })
    }
}
