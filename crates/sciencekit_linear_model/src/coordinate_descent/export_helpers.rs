//! Shared state encoding for the L1-regularized estimators built on the
//! coordinate-descent core.
//!
//! `SKLassoModel` and `SKElasticNetModel` differ only in the `l1_ratio` and
//! the tensor scope, so the codec lives here once and each model delegates.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKFloat, SKModelMetadata, SKModelState,
    SKParameterValue, SKTensor, sk_backend_parameter, sk_boolean_parameter, sk_float_parameter,
    sk_integer_parameter, sk_model_tensor_data_type, sk_parse_backend_parameter,
    sk_read_floating_tensor, sk_string_parameter, sk_validate_import_metadata,
};

use super::SKCoordinateDescentModelCore;
use super::accumulator::SKCoordinateDescentState;
use super::config::{SKCoordinateDescentConfig, SKSelection};

/// Encode the shared learned core under a family prefix.
pub(crate) fn export_coordinate_descent_state<F: SKFloat>(
    core: &SKCoordinateDescentModelCore<F>,
    prefix: &str,
    estimator_name: &str,
) -> Result<SKModelState<F>, SKError> {
    let feature_count = core.feature_count();
    let metadata = SKModelMetadata::new(
        estimator_name,
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        feature_count,
    )?
    .with_parameter(
        "include_intercept",
        SKParameterValue::from(core.include_intercept),
    )?
    .with_parameter(
        "tolerance",
        SKParameterValue::from(core.config.tolerance.to_f64().unwrap_or(1e-4)),
    )?
    .with_parameter(
        "maximum_number_of_iterations",
        SKParameterValue::from(core.config.maximum_number_of_iterations as i64),
    )?
    .with_parameter("positive", SKParameterValue::from(core.config.positive))?
    .with_parameter(
        "selection",
        SKParameterValue::from(selection_name(core.config.selection)),
    )?
    .with_parameter("screening", SKParameterValue::from(core.config.screening))?
    .with_parameter(
        "random_seed",
        SKParameterValue::from(core.config.random_seed as i64),
    )?
    .with_parameter(
        "screened_count",
        SKParameterValue::from(core.screened_count as i64),
    )?
    .with_parameter("backend", sk_backend_parameter(core.backend))?;
    let alphas = SKTensor::floating(
        format!("{prefix}.alphas"),
        vec![core.alphas.len()],
        core.alphas.clone(),
    )?;
    let coefficients = SKTensor::floating(
        format!("{prefix}.coefficients"),
        vec![core.coefficients.nrows(), feature_count],
        core.coefficients.iter().copied().collect(),
    )?;
    let intercepts = SKTensor::floating(
        format!("{prefix}.intercepts"),
        vec![core.intercepts.len()],
        core.intercepts.to_vec(),
    )?;
    let l1_ratio = SKTensor::floating(
        format!("{prefix}.l1_ratio"),
        Vec::new(),
        vec![core.l1_ratio],
    )?;
    SKModelState::new(metadata, vec![alphas, coefficients, intercepts, l1_ratio])
}

/// Rebuild the shared learned core from a family state.
pub(crate) fn import_coordinate_descent_core<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
    estimator_name: &str,
) -> Result<SKCoordinateDescentModelCore<F>, SKError> {
    let feature_count = state.metadata().feature_count();
    sk_validate_import_metadata(
        state,
        estimator_name,
        SK_MODEL_FORMAT_VERSION,
        feature_count,
    )?;
    let (_, alphas) = sk_read_floating_tensor(state, &format!("{prefix}.alphas"))?;
    let (coefficient_shape, coefficient_values) =
        sk_read_floating_tensor(state, &format!("{prefix}.coefficients"))?;
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
    let (_, intercepts) = sk_read_floating_tensor(state, &format!("{prefix}.intercepts"))?;
    let (_, l1_ratio_values) = sk_read_floating_tensor(state, &format!("{prefix}.l1_ratio"))?;
    if intercepts.len() != alpha_count || alphas.len() != alpha_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![alpha_count],
            found: vec![intercepts.len().max(alphas.len())],
        });
    }
    let &[l1_ratio] = l1_ratio_values.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![1],
            found: vec![l1_ratio_values.len()],
        });
    };
    let metadata = state.metadata();
    let config = SKCoordinateDescentConfig {
        tolerance: F::from(sk_float_parameter(metadata, "tolerance", 1e-4))
            .ok_or(SKError::Conversion("tolerance is not finite".to_string()))?,
        maximum_number_of_iterations: sk_integer_parameter(
            metadata,
            "maximum_number_of_iterations",
            1000,
        )
        .max(1) as usize,
        positive: sk_boolean_parameter(metadata, "positive", false),
        selection: selection_from_name(sk_string_parameter(metadata, "selection", "cyclic")),
        screening: sk_boolean_parameter(metadata, "screening", true),
        random_seed: sk_integer_parameter(metadata, "random_seed", 0).max(0) as u64,
    };
    Ok(SKCoordinateDescentModelCore {
        alphas,
        l1_ratio,
        coefficients,
        intercepts: Array1::from_vec(intercepts),
        include_intercept: sk_boolean_parameter(metadata, "include_intercept", true),
        backend: sk_parse_backend_parameter(metadata),
        execution_intent: SKExecutionMode::Automatic,
        screened_count: sk_integer_parameter(metadata, "screened_count", 0).max(0) as usize,
        config,
        state: SKCoordinateDescentState::new_dense(),
    })
}

/// Stable string for a coordinate sweep order.
fn selection_name(selection: SKSelection) -> &'static str {
    match selection {
        SKSelection::Cyclic => "cyclic",
        SKSelection::Random => "random",
    }
}

/// The sweep order for its stable string.
fn selection_from_name(name: &str) -> SKSelection {
    match name {
        "random" => SKSelection::Random,
        _ => SKSelection::Cyclic,
    }
}
