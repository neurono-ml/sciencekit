//! The registry that makes handles opaque non-dereferenceable tokens
//! (spec `ffi-abi`).
//!
//! A handle is a small integer cast to a pointer; the library never
//! dereferences it. Freeing removes the entry, so passing a freed handle back
//! returns [`SKFfiStatus::InvalidHandle`](crate::SKFfiStatus) without touching
//! memory. In-flight operations hold their own `Arc`, so a concurrent free
//! cannot pull the model out from under a prediction.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::builder::SKFfiBuilder;
use crate::model::SKFfiModel;
use crate::status::SKFfiStatus;

/// The payload behind one live handle.
pub(crate) enum SKFfiHandle {
    /// A mutable builder guarded for the setter calls.
    Builder(Arc<Mutex<SKFfiBuilder>>),
    /// A fitted model shared by concurrent readers.
    Model(Arc<SKFfiModel>),
}

/// A resolved, live handle. The `Arc` keeps the payload alive even when the
/// entry is freed concurrently.
pub(crate) enum SKFfiResolvedHandle {
    /// A live builder.
    Builder(Arc<Mutex<SKFfiBuilder>>),
    /// A live model.
    Model(Arc<SKFfiModel>),
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<SKFfiHandle>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<SKFfiHandle>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_identifier() -> u64 {
    static NEXT_IDENTIFIER: AtomicU64 = AtomicU64::new(1);
    loop {
        let identifier = NEXT_IDENTIFIER.fetch_add(1, Ordering::Relaxed);
        if identifier != 0 {
            return identifier;
        }
    }
}

/// Register a handle and return its opaque identifier.
pub(crate) fn sk_ffi_handle_insert(handle: SKFfiHandle) -> u64 {
    let identifier = next_identifier();
    let mut entries = registry()
        .lock()
        .expect("the handle registry is not poisoned");
    entries.insert(identifier, Arc::new(handle));
    identifier
}

/// Resolve a live handle by identifier, rejecting null, unknown and freed
/// identifiers.
pub(crate) fn sk_ffi_handle_lookup(identifier: u64) -> Result<SKFfiResolvedHandle, SKFfiStatus> {
    if identifier == 0 {
        return Err(SKFfiStatus::InvalidHandle);
    }
    let entries = registry()
        .lock()
        .expect("the handle registry is not poisoned");
    let handle = entries.get(&identifier).ok_or(SKFfiStatus::InvalidHandle)?;
    match handle.as_ref() {
        SKFfiHandle::Builder(builder) => Ok(SKFfiResolvedHandle::Builder(Arc::clone(builder))),
        SKFfiHandle::Model(model) => Ok(SKFfiResolvedHandle::Model(Arc::clone(model))),
    }
}

/// Remove a handle; freeing null or unknown identifiers is a no-op failure.
pub(crate) fn sk_ffi_handle_remove(identifier: u64) -> bool {
    if identifier == 0 {
        return false;
    }
    let mut entries = registry()
        .lock()
        .expect("the handle registry is not poisoned");
    entries.remove(&identifier).is_some()
}

#[cfg(test)]
#[path = "handle_tests.rs"]
mod handle_tests;
