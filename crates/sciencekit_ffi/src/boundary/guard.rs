//! Panic containment for every entry point (spec `ffi-abi`).
//!
//! No Rust unwind may cross the ABI: each wrapper catches a panic, records it
//! as [`SKFfiError::Internal`] and returns the matching status (or a null
//! pointer / default value).

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::error::{sk_ffi_clear_last_error, sk_ffi_record_error};
use crate::error_kind::SKFfiError;
use crate::status::SKFfiStatus;

/// Run a status-returning entry point, converting panics to `Internal`.
pub(crate) fn sk_ffi_guard_status<F>(body: F) -> SKFfiStatus
where
    F: FnOnce() -> Result<(), SKFfiError>,
{
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => {
            sk_ffi_clear_last_error();
            SKFfiStatus::Ok
        }
        Ok(Err(error)) => sk_ffi_record_error(&error),
        Err(payload) => sk_ffi_record_panic(payload),
    }
}

/// Run a pointer-returning entry point, returning null on failure.
pub(crate) fn sk_ffi_guard_pointer<T, F>(body: F) -> *mut T
where
    F: FnOnce() -> Result<*mut T, SKFfiError>,
{
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(pointer)) => {
            sk_ffi_clear_last_error();
            pointer
        }
        Ok(Err(error)) => {
            sk_ffi_record_error(&error);
            std::ptr::null_mut()
        }
        Err(payload) => {
            sk_ffi_record_panic(payload);
            std::ptr::null_mut()
        }
    }
}

/// Run a value-returning entry point, returning `T::default()` on failure.
pub(crate) fn sk_ffi_guard_value<T, F>(body: F) -> T
where
    T: Default,
    F: FnOnce() -> Result<T, SKFfiError>,
{
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => {
            sk_ffi_clear_last_error();
            value
        }
        Ok(Err(error)) => {
            sk_ffi_record_error(&error);
            T::default()
        }
        Err(payload) => {
            sk_ffi_record_panic(payload);
            T::default()
        }
    }
}

/// Turn a caught panic into an internal error, keeping its message when it is
/// textual.
fn sk_ffi_record_panic(payload: Box<dyn Any + Send>) -> SKFfiStatus {
    let message = if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-textual panic".to_string()
    };
    sk_ffi_record_error(&SKFfiError::Internal(format!("panic: {message}")))
}

#[cfg(test)]
#[path = "guard_tests.rs"]
mod guard_tests;
