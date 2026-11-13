//! Shared helpers for translating a neutral model state into graph nodes.

use sciencekit_common::{
    SKError, SKFloat, SKModelState, sk_boolean_parameter, sk_float_parameter, sk_integer_parameter,
    sk_read_floating_tensor, sk_read_integer_tensor, sk_string_parameter,
};

/// Read a floating tensor with its declared shape.
pub(crate) fn floating_tensor<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
) -> Result<(Vec<usize>, Vec<F>), SKError> {
    sk_read_floating_tensor(state, name)
}

/// Read an integer tensor.
#[allow(dead_code)]
pub(crate) fn integer_tensor<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
) -> Result<Vec<i64>, SKError> {
    sk_read_integer_tensor(state, name)
}

/// Read a floating parameter.
pub(crate) fn float_parameter<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
    default: f64,
) -> f64 {
    sk_float_parameter(state.metadata(), name, default)
}

/// Read an integer parameter.
#[allow(dead_code)]
pub(crate) fn integer_parameter<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
    default: i64,
) -> i64 {
    sk_integer_parameter(state.metadata(), name, default)
}

/// Read a boolean parameter.
pub(crate) fn boolean_parameter<F: SKFloat>(
    state: &SKModelState<F>,
    name: &str,
    default: bool,
) -> bool {
    sk_boolean_parameter(state.metadata(), name, default)
}

/// Read a string parameter.
pub(crate) fn string_parameter<'a, F: SKFloat>(
    state: &'a SKModelState<F>,
    name: &str,
    default: &'a str,
) -> &'a str {
    sk_string_parameter(state.metadata(), name, default)
}

/// Build the unsupported-family error, naming the estimator and the reason.
pub(crate) fn unsupported(family: &str, reason: impl Into<String>) -> SKError {
    SKError::UnsupportedOperation {
        operation: "onnx_export",
        reason: format!(
            "the estimator family `{family}` is not supported: {}",
            reason.into()
        ),
    }
}

/// Reject a state whose feature count disagrees with the stage input width.
#[allow(dead_code)]
pub(crate) fn require_feature_count<F: SKFloat>(
    state: &SKModelState<F>,
    expected: usize,
    family: &str,
) -> Result<(), SKError> {
    if state.metadata().feature_count() != expected {
        return Err(SKError::ShapeMismatch {
            expected: vec![expected],
            found: vec![state.metadata().feature_count()],
        });
    }
    let _ = family;
    Ok(())
}
