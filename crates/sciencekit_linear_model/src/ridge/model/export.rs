//! Format-agnostic state export/import for [`SKRidgeModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKParameterValue, SKTensor, sk_backend_parameter, sk_boolean_parameter,
    sk_integer_parameter, sk_model_tensor_data_type, sk_parse_backend_parameter,
    sk_read_floating_tensor, sk_string_parameter, sk_validate_import_metadata,
};

use super::SKRidgeModel;
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::gram::SKCenteredGramState;

/// Stable estimator identifier for exported ridge states.
pub const SK_RIDGE_ESTIMATOR_NAME: &str = "SKRidge";
/// The tensor carrying the configured penalties.
pub const SK_RIDGE_ALPHAS_TENSOR: &str = "ridge.alphas";
/// The tensor carrying one coefficient row per penalty.
pub const SK_RIDGE_COEFFICIENTS_TENSOR: &str = "ridge.coefficients";
/// The tensor carrying one intercept per penalty.
pub const SK_RIDGE_INTERCEPTS_TENSOR: &str = "ridge.intercepts";
/// The tensor carrying the design singular values.
pub const SK_RIDGE_SINGULAR_VALUES_TENSOR: &str = "ridge.singular_values";

impl<F: SKFloat> SKExportable<F> for SKRidgeModel<F> {
    fn export_state(&self) -> Result<sciencekit_common::SKModelState<F>, SKError> {
        let feature_count = self.coefficients.ncols();
        let metadata = SKModelMetadata::new(
            SK_RIDGE_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            feature_count,
        )?
        .with_parameter("rank", SKParameterValue::from(self.rank as i64))?
        .with_parameter(
            "include_intercept",
            SKParameterValue::from(self.include_intercept),
        )?
        .with_parameter("solver", SKParameterValue::from(solver_name(self.solver)))?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let alphas = SKTensor::floating(
            SK_RIDGE_ALPHAS_TENSOR,
            vec![self.alphas.len()],
            self.alphas.clone(),
        )?;
        let coefficients = SKTensor::floating(
            SK_RIDGE_COEFFICIENTS_TENSOR,
            vec![self.coefficients.nrows(), feature_count],
            self.coefficients.iter().copied().collect(),
        )?;
        let intercepts = SKTensor::floating(
            SK_RIDGE_INTERCEPTS_TENSOR,
            vec![self.intercepts.len()],
            self.intercepts.to_vec(),
        )?;
        let singular_values = SKTensor::floating(
            SK_RIDGE_SINGULAR_VALUES_TENSOR,
            vec![self.singular_values.len()],
            self.singular_values.clone(),
        )?;
        sciencekit_common::SKModelState::new(
            metadata,
            vec![alphas, coefficients, intercepts, singular_values],
        )
    }
}

impl<F: SKFloat> SKImportable<F> for SKRidgeModel<F> {
    fn import_state(state: &sciencekit_common::SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_RIDGE_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (_, alphas) = sk_read_floating_tensor(state, SK_RIDGE_ALPHAS_TENSOR)?;
        let (coefficient_shape, coefficient_values) =
            sk_read_floating_tensor(state, SK_RIDGE_COEFFICIENTS_TENSOR)?;
        let &[alpha_count, columns] = coefficient_shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: coefficient_shape,
            });
        };
        if columns != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![alpha_count, feature_count],
                found: vec![alpha_count, columns],
            });
        }
        let coefficients = Array2::from_shape_vec((alpha_count, columns), coefficient_values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let (_, intercepts) = sk_read_floating_tensor(state, SK_RIDGE_INTERCEPTS_TENSOR)?;
        if intercepts.len() != alpha_count || alphas.len() != alpha_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![alpha_count],
                found: vec![intercepts.len().max(alphas.len())],
            });
        }
        let (_, singular_values) = sk_read_floating_tensor(state, SK_RIDGE_SINGULAR_VALUES_TENSOR)?;
        let metadata = state.metadata();
        let include_intercept = sk_boolean_parameter(metadata, "include_intercept", true);
        Ok(SKRidgeModel {
            alphas,
            coefficients,
            intercepts: Array1::from_vec(intercepts),
            include_intercept,
            solver: solver_from_name(sk_string_parameter(metadata, "solver", "automatic")),
            rank: sk_integer_parameter(metadata, "rank", 0).max(0) as usize,
            singular_values,
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
            gram: SKCenteredGramState::empty(columns + usize::from(include_intercept)),
        })
    }
}

/// The stable string for a ridge solver.
fn solver_name(solver: SKRidgeSolver) -> &'static str {
    match solver {
        SKRidgeSolver::Automatic => "automatic",
        SKRidgeSolver::Cholesky => "cholesky",
        SKRidgeSolver::Svd => "svd",
        SKRidgeSolver::SparseCg => "sparse_cg",
    }
}

/// The solver for its stable string, defaulting to automatic.
fn solver_from_name(name: &str) -> SKRidgeSolver {
    match name {
        "cholesky" => SKRidgeSolver::Cholesky,
        "svd" => SKRidgeSolver::Svd,
        "sparse_cg" => SKRidgeSolver::SparseCg,
        _ => SKRidgeSolver::Automatic,
    }
}
