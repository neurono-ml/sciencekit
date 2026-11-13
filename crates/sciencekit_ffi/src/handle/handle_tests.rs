//! Tests for the opaque handle registry (spec `ffi-abi`).

use std::sync::{Arc, Mutex};

use super::{
    SKFfiHandle, SKFfiResolvedHandle, sk_ffi_handle_insert, sk_ffi_handle_lookup,
    sk_ffi_handle_remove,
};
use crate::array::SKFfiDataType;
use crate::builder::{SKFfiBuilder, SKFfiEstimatorFamily};
use crate::status::SKFfiStatus;

/// Create, resolve and free a builder handle.
#[test]
fn handle_create_and_free_round_trip() {
    let builder = Arc::new(Mutex::new(SKFfiBuilder::new(
        SKFfiEstimatorFamily::LinearRegression,
        SKFfiDataType::F64,
    )));
    let identifier = sk_ffi_handle_insert(SKFfiHandle::Builder(builder));
    assert!(identifier > 0);
    match sk_ffi_handle_lookup(identifier) {
        Ok(SKFfiResolvedHandle::Builder(resolved)) => assert_eq!(
            resolved.lock().unwrap().family(),
            SKFfiEstimatorFamily::LinearRegression
        ),
        _ => panic!("a builder handle must resolve to a builder"),
    }
    assert!(sk_ffi_handle_remove(identifier));
}

/// A null identifier is rejected without touching the registry.
#[test]
fn handle_null_is_rejected() {
    assert!(matches!(
        sk_ffi_handle_lookup(0),
        Err(SKFfiStatus::InvalidHandle)
    ));
    assert!(!sk_ffi_handle_remove(0));
}

/// A freed identifier resolves to an invalid-handle status, never a
/// dereference.
#[test]
fn handle_use_after_free_is_rejected() {
    let builder = Arc::new(Mutex::new(SKFfiBuilder::new(
        SKFfiEstimatorFamily::StandardScaler,
        SKFfiDataType::F64,
    )));
    let identifier = sk_ffi_handle_insert(SKFfiHandle::Builder(builder));
    assert!(sk_ffi_handle_remove(identifier));
    assert!(matches!(
        sk_ffi_handle_lookup(identifier),
        Err(SKFfiStatus::InvalidHandle)
    ));
    assert!(!sk_ffi_handle_remove(identifier));
}

/// Identifiers are never reused, so a stale handle cannot alias a new object.
#[test]
fn handle_identifiers_are_monotonic() {
    let first = sk_ffi_handle_insert(SKFfiHandle::Builder(Arc::new(Mutex::new(
        SKFfiBuilder::new(SKFfiEstimatorFamily::LinearRegression, SKFfiDataType::F64),
    ))));
    let second = sk_ffi_handle_insert(SKFfiHandle::Builder(Arc::new(Mutex::new(
        SKFfiBuilder::new(SKFfiEstimatorFamily::LinearRegression, SKFfiDataType::F64),
    ))));
    assert!(second > first);
    sk_ffi_handle_remove(first);
    sk_ffi_handle_remove(second);
}
