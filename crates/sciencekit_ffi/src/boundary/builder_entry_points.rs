//! Builder lifecycle and configuration entry points.

use std::sync::{Arc, Mutex};

use crate::boundary::guard::{sk_ffi_guard_pointer, sk_ffi_guard_status};
use crate::boundary::pointers::sk_ffi_handle_identifier;
use crate::builder::{SKFfiBuilder, SKFfiEstimatorFamily};
use crate::error_kind::SKFfiError;
use crate::handle::{
    SKFfiHandle, sk_ffi_handle_insert, sk_ffi_handle_lookup, sk_ffi_handle_remove,
};
use crate::status::SKFfiStatus;

/// Create a builder for one estimator family and scalar dtype.
///
/// Returns null on failure and records the reason in `sk_ffi_last_error`.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_builder_create(estimator_family: i32, dtype: i32) -> *mut SKFfiBuilder {
    sk_ffi_guard_pointer(|| {
        let family = SKFfiEstimatorFamily::from_code(estimator_family).ok_or_else(|| {
            SKFfiError::invalid_argument(format!(
                "unknown estimator family {estimator_family}; expected 0 (linear-regression) or 1 (standard-scaler)"
            ))
        })?;
        let dtype = crate::array::SKFfiDataType::from_code(dtype)
            .ok_or_else(|| SKFfiError::invalid_argument(format!("unknown dtype code {dtype}")))?;
        if !dtype.is_floating() {
            return Err(SKFfiError::Unsupported {
                operation: "builder_create",
                reason: format!("modeling requires a floating dtype, found {dtype:?}"),
            });
        }
        let builder = Arc::new(Mutex::new(SKFfiBuilder::new(family, dtype)));
        let identifier = sk_ffi_handle_insert(SKFfiHandle::Builder(builder));
        Ok(identifier as usize as *mut SKFfiBuilder)
    })
}

/// Set whether the linear model solves for an intercept.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_builder_set_include_intercept(
    builder: *mut SKFfiBuilder,
    enabled: bool,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let builder = sk_ffi_resolve_builder(builder)?;
        let mut configuration = builder.lock().map_err(|_| {
            SKFfiError::Internal("the builder lock was poisoned by an earlier panic".to_string())
        })?;
        configuration.set_include_intercept(enabled)
    })
}

/// Set whether the standard scaler centers its columns.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_builder_set_with_mean(
    builder: *mut SKFfiBuilder,
    enabled: bool,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let builder = sk_ffi_resolve_builder(builder)?;
        let mut configuration = builder.lock().map_err(|_| {
            SKFfiError::Internal("the builder lock was poisoned by an earlier panic".to_string())
        })?;
        configuration.set_with_mean(enabled)
    })
}

/// Set whether the standard scaler divides by column standard deviation.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_builder_set_with_std(
    builder: *mut SKFfiBuilder,
    enabled: bool,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let builder = sk_ffi_resolve_builder(builder)?;
        let mut configuration = builder.lock().map_err(|_| {
            SKFfiError::Internal("the builder lock was poisoned by an earlier panic".to_string())
        })?;
        configuration.set_with_std(enabled)
    })
}

/// Set the math backend kind from its stable code.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_builder_set_backend(
    builder: *mut SKFfiBuilder,
    backend: i32,
) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let builder = sk_ffi_resolve_builder(builder)?;
        let mut configuration = builder.lock().map_err(|_| {
            SKFfiError::Internal("the builder lock was poisoned by an earlier panic".to_string())
        })?;
        configuration.set_backend(backend)
    })
}

/// Free a builder. Null, unknown and already-freed handles are rejected.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_builder_free(builder: *mut SKFfiBuilder) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        let identifier = sk_ffi_handle_identifier(builder);
        match sk_ffi_handle_lookup(identifier) {
            Ok(crate::handle::SKFfiResolvedHandle::Builder(_)) => {}
            Ok(crate::handle::SKFfiResolvedHandle::Model(_)) => {
                return Err(SKFfiError::InvalidHandle);
            }
            Err(status) => return Err(SKFfiError::from_status(status)),
        }
        if sk_ffi_handle_remove(identifier) {
            Ok(())
        } else {
            Err(SKFfiError::InvalidHandle)
        }
    })
}

/// Resolve a builder handle, rejecting null, freed and wrong-kind pointers.
pub(crate) fn sk_ffi_resolve_builder(
    builder: *mut SKFfiBuilder,
) -> Result<Arc<Mutex<SKFfiBuilder>>, SKFfiError> {
    let identifier = sk_ffi_handle_identifier(builder);
    match sk_ffi_handle_lookup(identifier) {
        Ok(crate::handle::SKFfiResolvedHandle::Builder(configuration)) => Ok(configuration),
        Ok(crate::handle::SKFfiResolvedHandle::Model(_)) => Err(SKFfiError::InvalidHandle),
        Err(status) => Err(SKFfiError::from_status(status)),
    }
}
