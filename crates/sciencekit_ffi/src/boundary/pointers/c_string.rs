//! Handle-token and C-string conversions.

use std::ffi::{CStr, c_char};
use std::path::PathBuf;

use crate::error_kind::SKFfiError;

/// Turn a handle pointer into its registry identifier without dereferencing.
pub(crate) fn sk_ffi_handle_identifier<T>(handle: *mut T) -> u64 {
    handle as usize as u64
}

/// Read a NUL-terminated path from C.
///
/// # Safety
///
/// `path` must be null or point to a NUL-terminated readable string.
pub(crate) unsafe fn sk_ffi_path_from_c(path: *const c_char) -> Result<PathBuf, SKFfiError> {
    if path.is_null() {
        return Err(SKFfiError::invalid_argument("path pointer is null"));
    }
    // SAFETY: the caller guarantees a NUL-terminated string.
    let text = unsafe { CStr::from_ptr(path) }
        .to_str()
        .map_err(|_| SKFfiError::invalid_argument("path is not valid UTF-8"))?;
    Ok(PathBuf::from(text))
}
