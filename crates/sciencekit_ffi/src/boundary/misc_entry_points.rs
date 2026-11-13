//! Version, feature, status-name, last-error and test-hook entry points.

use std::ffi::c_char;
use std::panic::catch_unwind;

use crate::abi::{SK_FFI_ABI_MAJOR, SK_FFI_ABI_MINOR, SK_FFI_ABI_PATCH, SK_FFI_FEATURES};
use crate::boundary::guard::{sk_ffi_guard_status, sk_ffi_guard_value};
use crate::error::sk_ffi_last_error_pointer;
use crate::error_kind::SKFfiError;
use crate::status::SKFfiStatus;

/// Read the ABI version into the three out-pointers.
///
/// # Safety
///
/// Each non-null pointer must be writable for the duration of the call.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_version(major: *mut i32, minor: *mut i32, patch: *mut i32) -> SKFfiStatus {
    sk_ffi_guard_status(|| {
        if major.is_null() || minor.is_null() || patch.is_null() {
            return Err(SKFfiError::invalid_argument(
                "version out-pointers must not be null",
            ));
        }
        // SAFETY: the caller guarantees three writable pointers.
        unsafe {
            *major = SK_FFI_ABI_MAJOR;
            *minor = SK_FFI_ABI_MINOR;
            *patch = SK_FFI_ABI_PATCH;
        }
        Ok(())
    })
}

/// The feature bitmask the loaded library supports.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_features() -> u64 {
    sk_ffi_guard_value(|| Ok(SK_FFI_FEATURES))
}

/// The lowercase name of a status code, valid for the process lifetime.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_status_name(status: i32) -> *const c_char {
    catch_unwind(|| match SKFfiStatus::from_code(status) {
        Some(SKFfiStatus::Ok) => c"ok".as_ptr(),
        Some(SKFfiStatus::InvalidHandle) => c"invalid-handle".as_ptr(),
        Some(SKFfiStatus::Shape) => c"shape".as_ptr(),
        Some(SKFfiStatus::InvalidArgument) => c"invalid-argument".as_ptr(),
        Some(SKFfiStatus::Unsupported) => c"unsupported".as_ptr(),
        Some(SKFfiStatus::NotFitted) => c"not-fitted".as_ptr(),
        Some(SKFfiStatus::Io) => c"io".as_ptr(),
        Some(SKFfiStatus::Internal) => c"internal".as_ptr(),
        None => c"unknown".as_ptr(),
    })
    .unwrap_or(c"unknown".as_ptr())
}

/// The calling thread's last-error message; empty after a successful call.
///
/// The pointer stays valid until the next ABI call on the same thread. This
/// entry point deliberately does not clear the message it returns.
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_last_error() -> *const c_char {
    catch_unwind(sk_ffi_last_error_pointer)
        .unwrap_or(c"panic while reading the last error".as_ptr())
}

/// Test-only hook that proves panics never unwind across the ABI.
#[doc(hidden)]
#[cfg(any(test, feature = "test-hook"))]
#[unsafe(no_mangle)]
pub extern "C" fn sk_ffi_internal_test_panic() -> i32 {
    sk_ffi_guard_status(|| {
        panic!("deliberate panic from the sciencekit FFI test hook");
    }) as i32
}
