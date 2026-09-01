//! Convenience wrapper running an operation under an observability span
//! (spec `observability`).

use super::operation_attributes::SKOperationAttributes;
use super::operation_observation::SKOperationObservation;

/// Run an operation under a [`tracing`] span, recording errors and duration.
///
/// Convenience over [`SKOperationObservation`]: algorithm crates wrap each
/// public operation so failures stay visible in the trace.
pub fn sk_run_operation<T, E>(
    attributes: SKOperationAttributes,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E>
where
    E: std::error::Error,
{
    let observation = SKOperationObservation::begin(attributes);
    let result = observation.in_scope(operation);
    if let Err(error) = &result {
        observation.record_error(error);
    }
    observation.finish();
    result
}
