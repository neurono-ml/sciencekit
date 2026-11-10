//! The safe wrapper over an opaque fitted-model handle and its outputs.

use std::ffi::CString;
use std::ptr;

use sciencekit_ffi::SKFfiStatus;

use crate::array::{SKLuaMatrix, SKLuaVector};
use crate::engine::descriptor::SKLuaDescriptor;
use crate::engine::model::output::{sk_lua_empty_output, sk_lua_take_output};
use crate::error::SKLuaError;
use crate::lifecycle::{sk_lua_handle_created, sk_lua_handle_freed};

/// Owns one `SKFfiModel` handle and releases it exactly once.
///
/// The C ABI documents fitted models as immutable and safe to read from
/// several threads, and the handle registry keeps an `Arc` per operation, so
/// the wrapper is `Send + Sync` even though the raw pointer is not.
#[derive(Debug)]
pub(crate) struct SKLuaModelHandle {
    pointer: Option<*mut sciencekit_ffi::model::SKFfiModel>,
}

// SAFETY: every operation borrows the pointer immutably, and the C ABI
// resolves handles through a process-wide mutex-protected registry that hands
// each call its own `Arc`. Freeing takes `&mut self`, so it cannot race a
// borrowed read.
unsafe impl Send for SKLuaModelHandle {}
// SAFETY: as above; concurrent readers only observe the shared model.
unsafe impl Sync for SKLuaModelHandle {}

impl SKLuaModelHandle {
    /// Wrap a pointer the ABI just produced, rejecting null on error.
    pub(crate) fn from_fitted_pointer(
        pointer: *mut sciencekit_ffi::model::SKFfiModel,
    ) -> Result<SKLuaModelHandle, SKLuaError> {
        if pointer.is_null() {
            return Err(SKLuaError::from_last_error(SKFfiStatus::Internal));
        }
        sk_lua_handle_created();
        Ok(SKLuaModelHandle {
            pointer: Some(pointer),
        })
    }

    /// Load a fitted model from a path.
    pub(crate) fn load(path: &str) -> Result<SKLuaModelHandle, SKLuaError> {
        let path = sk_lua_c_string_path(path)?;
        let mut pointer: *mut sciencekit_ffi::model::SKFfiModel = ptr::null_mut();
        let status = sciencekit_ffi::sk_ffi_model_load(path.as_ptr(), &mut pointer);
        if status != SKFfiStatus::Ok {
            return Err(SKLuaError::from_last_error(status));
        }
        sk_lua_handle_created();
        Ok(SKLuaModelHandle {
            pointer: Some(pointer),
        })
    }

    /// The live pointer, or an invalid-handle error after `close`.
    fn require_pointer(&self) -> Result<*mut sciencekit_ffi::model::SKFfiModel, SKLuaError> {
        self.pointer
            .ok_or_else(|| SKLuaError::invalid_handle("the model was already closed"))
    }

    /// Whether the handle has been released.
    pub(crate) fn is_closed(&self) -> bool {
        self.pointer.is_none()
    }

    /// Predict one response per row.
    pub(crate) fn predict(&self, features: &SKLuaMatrix) -> Result<SKLuaMatrix, SKLuaError> {
        let features = SKLuaDescriptor::matrix(features);
        let features_array = features.as_ffi_array();
        let mut output = sk_lua_empty_output();
        let status =
            sciencekit_ffi::sk_ffi_predict(self.require_pointer()?, &features_array, &mut output);
        sk_lua_take_output(status, output)
    }

    /// Transform every row.
    pub(crate) fn transform(&self, features: &SKLuaMatrix) -> Result<SKLuaMatrix, SKLuaError> {
        let features = SKLuaDescriptor::matrix(features);
        let features_array = features.as_ffi_array();
        let mut output = sk_lua_empty_output();
        let status =
            sciencekit_ffi::sk_ffi_transform(self.require_pointer()?, &features_array, &mut output);
        sk_lua_take_output(status, output)
    }

    /// Score the model against targets.
    pub(crate) fn score(
        &self,
        features: &SKLuaMatrix,
        targets: &SKLuaVector,
    ) -> Result<f64, SKLuaError> {
        let features = SKLuaDescriptor::matrix(features);
        let targets = SKLuaDescriptor::vector(targets);
        let features_array = features.as_ffi_array();
        let targets_array = targets.as_ffi_array();
        let mut score = 0.0_f64;
        let status = sciencekit_ffi::sk_ffi_score(
            self.require_pointer()?,
            &features_array,
            &targets_array,
            &mut score,
        );
        if status == SKFfiStatus::Ok {
            Ok(score)
        } else {
            Err(SKLuaError::from_last_error(status))
        }
    }

    /// Save the fitted model to a path.
    pub(crate) fn save(&self, path: &str) -> Result<(), SKLuaError> {
        let path = sk_lua_c_string_path(path)?;
        let status = sciencekit_ffi::sk_ffi_model_save(self.require_pointer()?, path.as_ptr());
        crate::engine::builder::sk_lua_check(status)
    }

    /// Release the handle; calling `close` twice is a no-op.
    pub(crate) fn close(&mut self) {
        if let Some(pointer) = self.pointer.take() {
            let _ = sciencekit_ffi::sk_ffi_model_free(pointer);
            sk_lua_handle_freed();
        }
    }
}

impl Drop for SKLuaModelHandle {
    fn drop(&mut self) {
        self.close();
    }
}

/// Convert a Rust path into a NUL-terminated C string.
fn sk_lua_c_string_path(path: &str) -> Result<CString, SKLuaError> {
    CString::new(path)
        .map_err(|_| SKLuaError::invalid_argument("the path contains an interior NUL byte"))
}
