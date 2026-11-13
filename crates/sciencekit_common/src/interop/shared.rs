//! Shared conventions every family exporter follows.
//!
//! The module defines the single format version, the scalar-to-data_type
//! mapping and the readers/parsers model crates reuse, so a family exporter
//! never re-implements a validation rule. Every reader returns a structured
//! [`SKError`] naming the offending tensor.

use crate::SKBackendKind;
use crate::SKError;
use crate::interop::SKModelMetadata;
use crate::interop::{SKModelState, SKParameterValue, SKTensorDataType, SKTensorValues};
use crate::sk_float::SKFloat;

/// The single format version every family exporter writes.
///
/// A state is versioned as a whole: renaming a tensor, changing its shape or
/// changing its meaning requires bumping this constant, and every importer
/// rejects a version it does not understand before reading a tensor.
pub const SK_MODEL_FORMAT_VERSION: u32 = 1;

/// The tensor data_type matching the model scalar `F`.
pub fn sk_model_tensor_data_type<F: SKFloat>() -> SKTensorDataType {
    if size_of::<F>() == size_of::<f32>() {
        SKTensorDataType::F32
    } else {
        SKTensorDataType::F64
    }
}

/// Read a floating tensor as its shape and flat values.
///
/// # Errors
///
/// Returns [`SKError::InvalidHyperparameter`] when the tensor is missing and
/// [`SKError::UnsupportedRepresentation`] when it does not carry floating
/// values.
pub fn sk_read_floating_tensor<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
) -> Result<(Vec<usize>, Vec<F>), SKError> {
    let tensor = state.tensor(name).ok_or(SKError::InvalidHyperparameter {
        name: "tensor",
        reason: format!("the exported state does not carry `{name}`"),
    })?;
    match tensor.values() {
        SKTensorValues::Floating(values) => Ok((tensor.shape().to_vec(), values.clone())),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-floating tensor",
            suggestion: "export this tensor with floating values",
        }),
    }
}

/// Read an integer tensor as its flat values.
///
/// # Errors
///
/// Returns [`SKError::InvalidHyperparameter`] when the tensor is missing and
/// [`SKError::UnsupportedRepresentation`] when it does not carry integers.
pub fn sk_read_integer_tensor<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
) -> Result<Vec<i64>, SKError> {
    let tensor = state.tensor(name).ok_or(SKError::InvalidHyperparameter {
        name: "tensor",
        reason: format!("the exported state does not carry `{name}`"),
    })?;
    match tensor.values() {
        SKTensorValues::Integer(values) => Ok(values.clone()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-integer tensor",
            suggestion: "export this tensor with integer values",
        }),
    }
}

/// Read a boolean tensor as its flat values.
///
/// # Errors
///
/// Returns [`SKError::InvalidHyperparameter`] when the tensor is missing and
/// [`SKError::UnsupportedRepresentation`] when it does not carry booleans.
pub fn sk_read_boolean_tensor<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
) -> Result<Vec<bool>, SKError> {
    let tensor = state.tensor(name).ok_or(SKError::InvalidHyperparameter {
        name: "tensor",
        reason: format!("the exported state does not carry `{name}`"),
    })?;
    match tensor.values() {
        SKTensorValues::Boolean(values) => Ok(values.clone()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-boolean tensor",
            suggestion: "export this tensor with boolean values",
        }),
    }
}

/// The metadata parameter carrying a resolved backend kind.
pub fn sk_backend_parameter(backend: SKBackendKind) -> SKParameterValue {
    SKParameterValue::from(backend.to_string())
}

/// Read the resolved backend from metadata, defaulting to [`SKBackendKind::Faer`].
///
/// The reader accepts the documented display names and the historical `Debug`
/// spellings, so states written before the display-name switch still import.
pub fn sk_parse_backend_parameter(metadata: &SKModelMetadata) -> SKBackendKind {
    metadata
        .parameter("backend")
        .and_then(SKParameterValue::as_string)
        .map(sk_parse_backend_name)
        .unwrap_or(SKBackendKind::Faer)
}

/// Parse a persisted backend name, defaulting to `Faer` for unknown spellings.
pub fn sk_parse_backend_name(name: &str) -> SKBackendKind {
    name.parse::<SKBackendKind>().unwrap_or(match name {
        "Faer" => SKBackendKind::Faer,
        "MatrixMultiply" => SKBackendKind::MatrixMultiply,
        "NdArrayLinalg" => SKBackendKind::NdArrayLinalg,
        "OpenCl" => SKBackendKind::OpenCl,
        "Cuda" => SKBackendKind::Cuda,
        "Rocm" => SKBackendKind::Rocm,
        _ => SKBackendKind::Faer,
    })
}

/// Read an integer metadata parameter, defaulting when absent.
pub fn sk_integer_parameter(metadata: &SKModelMetadata, name: &str, default: i64) -> i64 {
    metadata
        .parameter(name)
        .and_then(SKParameterValue::as_integer)
        .unwrap_or(default)
}

/// Read a boolean metadata parameter, defaulting when absent.
pub fn sk_boolean_parameter(metadata: &SKModelMetadata, name: &str, default: bool) -> bool {
    metadata
        .parameter(name)
        .and_then(SKParameterValue::as_bool)
        .unwrap_or(default)
}

/// Read a floating metadata parameter, defaulting when absent.
pub fn sk_float_parameter(metadata: &SKModelMetadata, name: &str, default: f64) -> f64 {
    metadata
        .parameter(name)
        .and_then(SKParameterValue::as_float)
        .unwrap_or(default)
}

/// Read a string metadata parameter, defaulting when absent.
pub fn sk_string_parameter<'a>(
    metadata: &'a SKModelMetadata,
    name: &str,
    default: &'a str,
) -> &'a str {
    metadata
        .parameter(name)
        .and_then(SKParameterValue::as_string)
        .unwrap_or(default)
}
