//! Format-agnostic state export/import for [`SKPCAModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKBuilderState, SKError, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_model_tensor_data_type, sk_parse_backend_parameter,
    sk_read_floating_tensor, sk_string_parameter, sk_validate_import_metadata,
};

use super::SKPCAModel;
use super::config::SKPCASolver;

/// Stable estimator identifier for exported PCA states.
pub const SK_PCA_ESTIMATOR_NAME: &str = "SKPCA";
/// The tensor carrying the principal components.
pub const SK_PCA_COMPONENTS_TENSOR: &str = "pca.components";
/// The tensor carrying the per-feature mean.
pub const SK_PCA_MEAN_TENSOR: &str = "pca.mean";
/// The tensor carrying the explained variance.
pub const SK_PCA_EXPLAINED_VARIANCE_TENSOR: &str = "pca.explained_variance";
/// The tensor carrying the explained-variance ratio.
pub const SK_PCA_EXPLAINED_VARIANCE_RATIO_TENSOR: &str = "pca.explained_variance_ratio";
/// The tensor carrying the singular values.
pub const SK_PCA_SINGULAR_VALUES_TENSOR: &str = "pca.singular_values";
/// The tensor carrying the whitening scales.
pub const SK_PCA_WHITEN_SCALE_TENSOR: &str = "pca.whiten_scale";

impl<F: SKFloat> SKExportable<F> for SKPCAModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_PCA_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.mean.len(),
        )?
        .with_parameter("whiten", SKParameterValue::from(self.whiten))?
        .with_parameter(
            "resolved_solver",
            SKParameterValue::from(solver_name(self.resolved_solver)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let tensors = vec![
            SKTensor::floating(
                SK_PCA_COMPONENTS_TENSOR,
                vec![self.components.nrows(), self.components.ncols()],
                self.components.iter().copied().collect(),
            )?,
            SKTensor::floating(
                SK_PCA_MEAN_TENSOR,
                vec![self.mean.len()],
                self.mean.to_vec(),
            )?,
            SKTensor::floating(
                SK_PCA_EXPLAINED_VARIANCE_TENSOR,
                vec![self.explained_variance.len()],
                self.explained_variance.to_vec(),
            )?,
            SKTensor::floating(
                SK_PCA_EXPLAINED_VARIANCE_RATIO_TENSOR,
                vec![self.explained_variance_ratio.len()],
                self.explained_variance_ratio.to_vec(),
            )?,
            SKTensor::floating(
                SK_PCA_SINGULAR_VALUES_TENSOR,
                vec![self.singular_values.len()],
                self.singular_values.to_vec(),
            )?,
            SKTensor::floating(
                SK_PCA_WHITEN_SCALE_TENSOR,
                vec![self.whiten_scale.len()],
                self.whiten_scale.to_vec(),
            )?,
        ];
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKPCAModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_PCA_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (component_shape, component_values) =
            sk_read_floating_tensor(state, SK_PCA_COMPONENTS_TENSOR)?;
        let &[components, columns] = component_shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: component_shape,
            });
        };
        if columns != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![components, feature_count],
                found: vec![components, columns],
            });
        }
        let components = Array2::from_shape_vec((components, columns), component_values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let (_, mean) = sk_read_floating_tensor(state, SK_PCA_MEAN_TENSOR)?;
        let (_, explained_variance) =
            sk_read_floating_tensor(state, SK_PCA_EXPLAINED_VARIANCE_TENSOR)?;
        let (_, explained_variance_ratio) =
            sk_read_floating_tensor(state, SK_PCA_EXPLAINED_VARIANCE_RATIO_TENSOR)?;
        let (_, singular_values) = sk_read_floating_tensor(state, SK_PCA_SINGULAR_VALUES_TENSOR)?;
        let (_, whiten_scale) = sk_read_floating_tensor(state, SK_PCA_WHITEN_SCALE_TENSOR)?;
        let metadata = state.metadata();
        Ok(SKPCAModel {
            components,
            explained_variance: Array1::from_vec(explained_variance),
            explained_variance_ratio: Array1::from_vec(explained_variance_ratio),
            singular_values: Array1::from_vec(singular_values),
            mean: Array1::from_vec(mean),
            whiten_scale: Array1::from_vec(whiten_scale),
            whiten: sk_boolean_parameter(metadata, "whiten", false),
            resolved_solver: solver_from_name(sk_string_parameter(
                metadata,
                "resolved_solver",
                "full",
            )),
            backend: sk_parse_backend_parameter(metadata),
            builder_state: SKBuilderState::new(),
        })
    }
}

fn solver_name(solver: SKPCASolver) -> &'static str {
    match solver {
        SKPCASolver::Auto => "auto",
        SKPCASolver::Full => "full",
        SKPCASolver::Randomized => "randomized",
        SKPCASolver::CovarianceEigh => "covariance_eigh",
    }
}

fn solver_from_name(name: &str) -> SKPCASolver {
    match name {
        "auto" => SKPCASolver::Auto,
        "randomized" => SKPCASolver::Randomized,
        "covariance_eigh" => SKPCASolver::CovarianceEigh,
        _ => SKPCASolver::Full,
    }
}
