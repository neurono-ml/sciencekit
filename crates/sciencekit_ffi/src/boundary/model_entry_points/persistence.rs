//! Model save and load entry points.

use std::sync::Arc;

use crate::boundary::guard::sk_ffi_guard_status;
use crate::boundary::model_entry_points::handles::sk_ffi_resolve_model;
use crate::boundary::pointers::sk_ffi_path_from_c;
use crate::error_kind::SKFfiError;
use crate::handle::{SKFfiHandle, sk_ffi_handle_insert};
use crate::model::{SKFfiModel, sk_ffi_load_model_file, sk_ffi_save_model_file};
use crate::status::SKFfiStatus;

/// Save a fitted model to a file path.
///
/// # Safety
///
/// The model must be a live handle and `path` a NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_model_save(
    model: *mut SKFfiModel,
    path: *const std::ffi::c_char,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let model = sk_ffi_resolve_model(model)?;
        // SAFETY: the caller guarantees a NUL-terminated path.
        let path = unsafe { sk_ffi_path_from_c(path)? };
        sk_ffi_save_model_file(&model, &path)
    })
}

/// Load a fitted model from a file path into `output`.
///
/// # Safety
///
/// `path` must be NUL-terminated and `output` writable.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_model_load(
    path: *const std::ffi::c_char,
    output: *mut *mut SKFfiModel,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        if output.is_null() {
            return Err(SKFfiError::invalid_argument("model out-pointer is null"));
        }
        // SAFETY: the caller guarantees a NUL-terminated path.
        let path = unsafe { sk_ffi_path_from_c(path)? };
        let model = sk_ffi_load_model_file(&path)?;
        let identifier = sk_ffi_handle_insert(SKFfiHandle::Model(Arc::new(model)));
        // SAFETY: the caller guarantees a writable out-pointer.
        unsafe { *output = identifier as usize as *mut SKFfiModel };
        Ok(())
    })
}
