//! Shared checks for the typed kernel operations (spec `opencl-backend`).

use ndarray::Array2;
use sciencekit_common::{SKError, SKFloat};

use crate::kernel::SKDeviceBufferRef;

use crate::opencl::session::SKOpenClSession;

/// Reject a scalar type the selected device does not execute.
pub(crate) fn ensure_scalar_supported<F: SKFloat>(
    session: &SKOpenClSession,
) -> Result<(), SKError> {
    if session.device().supports_scalar_type(F::SCALAR_NAME) {
        Ok(())
    } else {
        Err(SKError::UnsupportedScalarType {
            backend: "opencl",
            scalar_type: F::SCALAR_NAME,
        })
    }
}

/// The bytes `elements` scalar values occupy.
pub(crate) fn element_bytes<F: SKFloat>(elements: usize) -> u64 {
    (elements as u64).saturating_mul(std::mem::size_of::<F>() as u64)
}

/// Materialise one dispatch argument as an owned host matrix.
pub(crate) fn argument_matrix<F: SKFloat>(
    arguments: &[SKDeviceBufferRef<'_, F>],
    index: usize,
    operation: &'static str,
) -> Result<Array2<F>, SKError> {
    arguments
        .get(index)
        .ok_or_else(|| SKError::ShapeMismatch {
            expected: vec![index + 1],
            found: vec![arguments.len()],
        })?
        .to_host_matrix()
        .map_err(|error| SKError::UnsupportedOperation {
            operation,
            reason: format!("the argument matrix could not be read: {error}"),
        })
}
