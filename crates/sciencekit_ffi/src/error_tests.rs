//! Tests for the thread-local last-error message (spec `ffi-abi`).

use super::{sk_ffi_clear_last_error, sk_ffi_last_error_message, sk_ffi_set_last_error};
use crate::error_kind::SKFfiError;
use crate::status::SKFfiStatus;

/// Setting a message makes it readable on the same thread.
#[test]
fn error_success_clears_the_message() {
    sk_ffi_set_last_error("boom");
    assert_eq!(sk_ffi_last_error_message(), "boom");
    sk_ffi_clear_last_error();
    assert_eq!(sk_ffi_last_error_message(), "");
}

/// The buffer is per-thread: one thread's error never leaks to another.
#[test]
fn error_is_thread_local() {
    sk_ffi_set_last_error("main thread failure");
    let other = std::thread::spawn(sk_ffi_last_error_message)
        .join()
        .expect("the observing thread must not panic");
    assert_eq!(other, "");
    assert_eq!(sk_ffi_last_error_message(), "main thread failure");
    sk_ffi_clear_last_error();
}

/// A recorded failure reports its status and stores its rendered message.
#[test]
fn error_recording_maps_status_and_message() {
    let status = super::sk_ffi_record_error(&SKFfiError::InvalidHandle);
    assert_eq!(status, SKFfiStatus::InvalidHandle);
    assert!(sk_ffi_last_error_message().contains("invalid handle"));
    sk_ffi_clear_last_error();
}
