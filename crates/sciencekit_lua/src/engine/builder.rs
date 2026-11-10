//! The safe wrapper over an opaque builder handle.

use std::ptr;

use sciencekit_ffi::{SKFfiDataType, SKFfiStatus};

use crate::array::{SKLuaMatrix, SKLuaVector};
use crate::engine::descriptor::SKLuaDescriptor;
use crate::engine::family::SKLuaEstimatorFamily;
use crate::engine::model::SKLuaModelHandle;
use crate::error::SKLuaError;
use crate::lifecycle::{sk_lua_handle_created, sk_lua_handle_freed};

/// Owns one `SKFfiBuilder` handle and releases it exactly once.
#[derive(Debug)]
pub(crate) struct SKLuaBuilderHandle {
    pointer: Option<*mut sciencekit_ffi::builder::SKFfiBuilder>,
}

impl SKLuaBuilderHandle {
    /// Create an `f64` builder for one family.
    ///
    /// Lua 5.4 numbers are IEEE-754 doubles, so the binding models in `f64`
    /// only; `f32` descriptors would add rounding without saving memory.
    pub(crate) fn create(family: SKLuaEstimatorFamily) -> Result<SKLuaBuilderHandle, SKLuaError> {
        let pointer =
            sciencekit_ffi::sk_ffi_builder_create(family.code(), SKFfiDataType::F64 as i32);
        if pointer.is_null() {
            return Err(SKLuaError::from_last_error(SKFfiStatus::InvalidArgument));
        }
        sk_lua_handle_created();
        Ok(SKLuaBuilderHandle {
            pointer: Some(pointer),
        })
    }

    /// The live pointer, or an invalid-handle error after `close`.
    fn require_pointer(&self) -> Result<*mut sciencekit_ffi::builder::SKFfiBuilder, SKLuaError> {
        self.pointer
            .ok_or_else(|| SKLuaError::invalid_handle("the builder was already closed"))
    }

    /// Whether the handle has been released.
    pub(crate) fn is_closed(&self) -> bool {
        self.pointer.is_none()
    }

    /// Set whether the linear model solves for an intercept.
    pub(crate) fn set_include_intercept(&mut self, enabled: bool) -> Result<(), SKLuaError> {
        let status =
            sciencekit_ffi::sk_ffi_builder_set_include_intercept(self.require_pointer()?, enabled);
        sk_lua_check(status)
    }

    /// Set whether the scaler centers its columns.
    pub(crate) fn set_with_mean(&mut self, enabled: bool) -> Result<(), SKLuaError> {
        let status = sciencekit_ffi::sk_ffi_builder_set_with_mean(self.require_pointer()?, enabled);
        sk_lua_check(status)
    }

    /// Set whether the scaler divides by column standard deviation.
    pub(crate) fn set_with_std(&mut self, enabled: bool) -> Result<(), SKLuaError> {
        let status = sciencekit_ffi::sk_ffi_builder_set_with_std(self.require_pointer()?, enabled);
        sk_lua_check(status)
    }

    /// Set the math backend by its stable code.
    pub(crate) fn set_backend(&mut self, code: i32) -> Result<(), SKLuaError> {
        let status = sciencekit_ffi::sk_ffi_builder_set_backend(self.require_pointer()?, code);
        sk_lua_check(status)
    }

    /// Fit the configured estimator, returning a model handle.
    pub(crate) fn fit(
        &self,
        features: &SKLuaMatrix,
        targets: Option<&SKLuaVector>,
    ) -> Result<SKLuaModelHandle, SKLuaError> {
        let features = SKLuaDescriptor::matrix(features);
        let targets = targets.map(SKLuaDescriptor::vector);
        let features_array = features.as_ffi_array();
        let targets_array = targets.as_ref().map(SKLuaDescriptor::as_ffi_array);
        let targets_pointer = match &targets_array {
            Some(array) => array as *const _,
            None => ptr::null(),
        };
        let pointer =
            sciencekit_ffi::sk_ffi_fit(self.require_pointer()?, &features_array, targets_pointer);
        SKLuaModelHandle::from_fitted_pointer(pointer)
    }

    /// Release the handle; calling `close` twice is a no-op.
    pub(crate) fn close(&mut self) {
        if let Some(pointer) = self.pointer.take() {
            let _ = sciencekit_ffi::sk_ffi_builder_free(pointer);
            sk_lua_handle_freed();
        }
    }
}

impl Drop for SKLuaBuilderHandle {
    fn drop(&mut self) {
        self.close();
    }
}

/// Turn a non-OK ABI status into a structured error.
pub(crate) fn sk_lua_check(status: SKFfiStatus) -> Result<(), SKLuaError> {
    if status == SKFfiStatus::Ok {
        Ok(())
    } else {
        Err(SKLuaError::from_last_error(status))
    }
}
