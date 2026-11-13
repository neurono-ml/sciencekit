//! Small conversions shared by the affine mappers.

use sciencekit_common::{SKError, SKFloat, SKTensor};
use sciencekit_interop::native::SKNativeImportError;

use crate::native::error::SKOnnxNativeError;

pub(crate) fn tensor<F: SKFloat>(
    name: &str,
    shape: Vec<usize>,
    values: Vec<F>,
) -> Result<SKTensor<F>, SKOnnxNativeError> {
    SKTensor::floating(name, shape, values).map_err(native)
}

pub(crate) fn invalid(detail: impl Into<String>) -> SKOnnxNativeError {
    SKOnnxNativeError::InvalidGraph {
        detail: detail.into(),
    }
}

pub(crate) fn native(error: SKError) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(SKNativeImportError::from(error))
}
