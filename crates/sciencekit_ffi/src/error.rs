//! The thread-local last-error message (spec `ffi-abi`).
//!
//! A successful call clears the message; a failing call stores it. The
//! pointer [`sk_ffi_last_error_pointer`] returns stays valid until the next
//! call on the same thread, so bindings copy the string before handing it to
//! another thread.

use std::cell::RefCell;
use std::ffi::{CString, c_char};

use crate::error_kind::SKFfiError;

thread_local! {
    static LAST_ERROR: RefCell<CString> =
        RefCell::new(CString::new("").expect("an empty message has no interior nul"));
}

/// Replace the calling thread's last-error message.
///
/// Interior nul bytes are replaced with spaces so the C string stays valid.
pub fn sk_ffi_set_last_error(message: &str) {
    let sanitized = message.replace('\0', " ");
    let c_string = CString::new(sanitized)
        .unwrap_or_else(|_| CString::new("error message unavailable").expect("static message"));
    LAST_ERROR.with(|slot| *slot.borrow_mut() = c_string);
}

/// Clear the calling thread's last-error message.
pub fn sk_ffi_clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new("").expect("an empty message has no interior nul");
    });
}

/// Record a failure and return its status.
pub fn sk_ffi_record_error(error: &SKFfiError) -> crate::SKFfiStatus {
    sk_ffi_set_last_error(&error.to_string());
    error.status()
}

/// The calling thread's last-error message as a borrowed C string.
pub fn sk_ffi_last_error_pointer() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// The calling thread's last-error message, copied for Rust-side tests.
pub fn sk_ffi_last_error_message() -> String {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .to_str()
            .unwrap_or("error message is not valid UTF-8")
            .to_string()
    })
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod error_tests;
