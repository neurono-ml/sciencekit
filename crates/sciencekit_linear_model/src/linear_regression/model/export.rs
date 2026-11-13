//! Export and import of the fitted linear-regression state through the
//! format-agnostic `SKModelState` contract (spec `model-export-contracts`).

use ndarray::Array1;
use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable, SKModelMetadata,
    SKModelState, SKParameterValue, SKTensor, SKTensorValues, sk_validate_import_metadata,
};

use super::core_implementation::SKLinearRegressionModel;
use crate::linear_regression::gram::SKGramState;

/// The stable estimator name every exported linear-regression state carries.
pub const SK_LINEAR_REGRESSION_ESTIMATOR_NAME: &str = "SKLinearRegression";
/// The format version of the exported linear-regression state.
pub const SK_LINEAR_REGRESSION_FORMAT_VERSION: u32 = 1;
/// The stable tensor name of the coefficients.
pub const SK_LINEAR_COEFFICIENTS_TENSOR: &str = "linear.coefficients";
/// The stable tensor name of the intercept.
pub const SK_LINEAR_INTERCEPT_TENSOR: &str = "linear.intercept";

impl<F: SKFloat> SKExportable<F> for SKLinearRegressionModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_LINEAR_REGRESSION_ESTIMATOR_NAME,
            SK_LINEAR_REGRESSION_FORMAT_VERSION,
            tensor_dtype::<F>(),
            self.coefficients.len(),
        )?
        .with_parameter(
            "include_intercept",
            SKParameterValue::from(self.include_intercept),
        )?
        .with_parameter(
            "backend",
            SKParameterValue::from(format!("{:?}", self.backend)),
        )?;
        let coefficients = SKTensor::floating(
            SK_LINEAR_COEFFICIENTS_TENSOR,
            vec![self.coefficients.len()],
            self.coefficients.to_vec(),
        )?;
        let intercept =
            SKTensor::floating(SK_LINEAR_INTERCEPT_TENSOR, Vec::new(), vec![self.intercept])?;
        SKModelState::new(metadata, vec![coefficients, intercept])
    }
}

impl<F: SKFloat> SKImportable<F> for SKLinearRegressionModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        sk_validate_import_metadata(
            state,
            SK_LINEAR_REGRESSION_ESTIMATOR_NAME,
            SK_LINEAR_REGRESSION_FORMAT_VERSION,
            state.metadata().feature_count(),
        )?;
        let coefficients = floating_values(state, SK_LINEAR_COEFFICIENTS_TENSOR)?;
        let intercept = floating_values(state, SK_LINEAR_INTERCEPT_TENSOR)?;
        if coefficients.len() != state.metadata().feature_count() {
            return Err(SKError::ShapeMismatch {
                expected: vec![state.metadata().feature_count()],
                found: vec![coefficients.len()],
            });
        }
        let [intercept] = intercept.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1],
                found: vec![intercept.len()],
            });
        };
        let include_intercept = state
            .metadata()
            .parameter("include_intercept")
            .and_then(SKParameterValue::as_bool)
            .unwrap_or(true);
        let backend = state
            .metadata()
            .parameter("backend")
            .and_then(SKParameterValue::as_string)
            .map(parse_backend)
            .unwrap_or(SKBackendKind::Faer);
        Ok(restore(
            coefficients,
            *intercept,
            include_intercept,
            backend,
        ))
    }
}

/// Rebuild a fitted model from imported coefficients and the intercept.
fn restore<F: SKFloat>(
    coefficients: Vec<F>,
    intercept: F,
    include_intercept: bool,
    backend: SKBackendKind,
) -> SKLinearRegressionModel<F> {
    let design_width = coefficients.len() + usize::from(include_intercept);
    SKLinearRegressionModel {
        coefficients: Array1::from_vec(coefficients),
        intercept,
        rank: 0,
        singular_values: Vec::new(),
        gram: SKGramState::empty(design_width),
        include_intercept,
        backend,
        execution_intent: SKExecutionMode::Automatic,
    }
}

/// Read a floating tensor's values or report a malformed state.
fn floating_values<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
) -> Result<Vec<F>, SKError> {
    let tensor = state.tensor(name).ok_or(SKError::InvalidHyperparameter {
        name,
        reason: "the exported state does not carry this tensor".to_string(),
    })?;
    match tensor.values() {
        SKTensorValues::Floating(values) => Ok(values.clone()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-floating tensor",
            suggestion: "export this tensor with floating values",
        }),
    }
}

/// The tensor dtype matching the model scalar.
fn tensor_dtype<F: SKFloat>() -> sciencekit_common::SKTensorDataType {
    if size_of::<F>() == size_of::<f32>() {
        sciencekit_common::SKTensorDataType::F32
    } else {
        sciencekit_common::SKTensorDataType::F64
    }
}

/// Parse the persisted backend name, accepting both `Debug` and lowercase
/// [`SKBackendKind::name`] spellings and defaulting to `Faer` for unknowns.
fn parse_backend(name: &str) -> SKBackendKind {
    if let Ok(kind) = name.parse::<SKBackendKind>() {
        return kind;
    }
    match name {
        "Faer" => SKBackendKind::Faer,
        "MatrixMultiply" => SKBackendKind::MatrixMultiply,
        "NdArrayLinalg" => SKBackendKind::NdArrayLinalg,
        "OpenCl" => SKBackendKind::OpenCl,
        "Cuda" => SKBackendKind::Cuda,
        "Rocm" => SKBackendKind::Rocm,
        _ => SKBackendKind::Faer,
    }
}
