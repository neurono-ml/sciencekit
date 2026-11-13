//! Format-agnostic state export/import for [`SKTruncatedSVDModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKBuilderState, SKError, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_model_tensor_data_type, sk_parse_backend_parameter, sk_read_floating_tensor,
    sk_string_parameter, sk_validate_import_metadata,
};

use super::SKTruncatedSVDModel;
use super::config::SKTruncatedSVDAlgorithm;

/// Stable estimator identifier for exported truncated-SVD states.
pub const SK_TRUNCATED_SVD_ESTIMATOR_NAME: &str = "SKTruncatedSVD";
/// The tensor carrying the right singular vectors.
pub const SK_TRUNCATED_SVD_COMPONENTS_TENSOR: &str = "truncated_svd.components";
/// The tensor carrying the retained singular values.
pub const SK_TRUNCATED_SVD_SINGULAR_VALUES_TENSOR: &str = "truncated_svd.singular_values";
/// The tensor carrying the explained variance.
pub const SK_TRUNCATED_SVD_EXPLAINED_VARIANCE_TENSOR: &str = "truncated_svd.explained_variance";
/// The tensor carrying the explained-variance ratio.
pub const SK_TRUNCATED_SVD_RATIO_TENSOR: &str = "truncated_svd.explained_variance_ratio";

impl<F: SKFloat> SKExportable<F> for SKTruncatedSVDModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_TRUNCATED_SVD_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.components.ncols(),
        )?
        .with_parameter(
            "algorithm",
            SKParameterValue::from(algorithm_name(self.algorithm)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let tensors = vec![
            SKTensor::floating(
                SK_TRUNCATED_SVD_COMPONENTS_TENSOR,
                vec![self.components.nrows(), self.components.ncols()],
                self.components.iter().copied().collect(),
            )?,
            SKTensor::floating(
                SK_TRUNCATED_SVD_SINGULAR_VALUES_TENSOR,
                vec![self.singular_values.len()],
                self.singular_values.to_vec(),
            )?,
            SKTensor::floating(
                SK_TRUNCATED_SVD_EXPLAINED_VARIANCE_TENSOR,
                vec![self.explained_variance.len()],
                self.explained_variance.to_vec(),
            )?,
            SKTensor::floating(
                SK_TRUNCATED_SVD_RATIO_TENSOR,
                vec![self.explained_variance_ratio.len()],
                self.explained_variance_ratio.to_vec(),
            )?,
        ];
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKTruncatedSVDModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_TRUNCATED_SVD_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (shape, values) = sk_read_floating_tensor(state, SK_TRUNCATED_SVD_COMPONENTS_TENSOR)?;
        let &[components, columns] = shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: shape,
            });
        };
        if columns != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![components, feature_count],
                found: vec![components, columns],
            });
        }
        let components = Array2::from_shape_vec((components, columns), values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let (_, singular_values) =
            sk_read_floating_tensor(state, SK_TRUNCATED_SVD_SINGULAR_VALUES_TENSOR)?;
        let (_, explained_variance) =
            sk_read_floating_tensor(state, SK_TRUNCATED_SVD_EXPLAINED_VARIANCE_TENSOR)?;
        let (_, explained_variance_ratio) =
            sk_read_floating_tensor(state, SK_TRUNCATED_SVD_RATIO_TENSOR)?;
        sk_require_length(
            &singular_values,
            components.nrows(),
            SK_TRUNCATED_SVD_SINGULAR_VALUES_TENSOR,
        )?;
        sk_require_length(
            &explained_variance,
            components.nrows(),
            SK_TRUNCATED_SVD_EXPLAINED_VARIANCE_TENSOR,
        )?;
        sk_require_length(
            &explained_variance_ratio,
            components.nrows(),
            SK_TRUNCATED_SVD_RATIO_TENSOR,
        )?;
        let metadata = state.metadata();
        Ok(SKTruncatedSVDModel {
            components,
            singular_values: Array1::from_vec(singular_values),
            explained_variance: Array1::from_vec(explained_variance),
            explained_variance_ratio: Array1::from_vec(explained_variance_ratio),
            algorithm: algorithm_from_name(sk_string_parameter(
                metadata,
                "algorithm",
                "randomized",
            )),
            backend: sk_parse_backend_parameter(metadata),
            builder_state: SKBuilderState::new(),
        })
    }
}

fn algorithm_name(algorithm: SKTruncatedSVDAlgorithm) -> &'static str {
    match algorithm {
        SKTruncatedSVDAlgorithm::Randomized => "randomized",
        SKTruncatedSVDAlgorithm::Arpack => "arpack",
    }
}

fn algorithm_from_name(name: &str) -> SKTruncatedSVDAlgorithm {
    match name {
        "arpack" => SKTruncatedSVDAlgorithm::Arpack,
        _ => SKTruncatedSVDAlgorithm::Randomized,
    }
}

/// Reject a secondary tensor whose length disagrees with the component count,
/// so a malformed state never reaches a broadcasting panic.
fn sk_require_length<F: SKFloat>(values: &[F], expected: usize, name: &str) -> Result<(), SKError> {
    if values.len() != expected {
        return Err(SKError::ShapeMismatch {
            expected: vec![expected],
            found: vec![values.len()],
        });
    }
    let _ = name;
    Ok(())
}
