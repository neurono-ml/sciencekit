//! Tests that a panic never unwinds across the ABI (spec `ffi-abi`).

use crate::boundary::sk_ffi_internal_test_panic;
use crate::error::sk_ffi_last_error_message;
use crate::status::SKFfiStatus;

/// A deliberate panic becomes an internal-error status and the process stays
/// alive.
#[test]
fn panic_is_converted_to_internal_status() {
    let status = sk_ffi_internal_test_panic();
    assert_eq!(status, SKFfiStatus::Internal as i32);
    let message = sk_ffi_last_error_message();
    assert!(
        message.contains("panic"),
        "the recorded message must explain the panic: {message}"
    );
    assert!(message.contains("deliberate panic"));
}

/// After a caught panic, the next successful call clears the message.
#[test]
fn panic_recovery_clears_the_error_on_success() {
    assert_eq!(sk_ffi_internal_test_panic(), SKFfiStatus::Internal as i32);
    let mut major = 0_i32;
    let mut minor = 0_i32;
    let mut patch = 0_i32;
    assert_eq!(
        crate::boundary::sk_ffi_version(&mut major, &mut minor, &mut patch),
        SKFfiStatus::Ok
    );
    assert_eq!(sk_ffi_last_error_message(), "");
}
