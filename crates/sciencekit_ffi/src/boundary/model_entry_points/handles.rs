//! Resolving a model handle without dereferencing the caller's pointer.

use std::sync::Arc;

use crate::boundary::pointers::sk_ffi_handle_identifier;
use crate::error_kind::SKFfiError;
use crate::handle::sk_ffi_handle_lookup;
use crate::model::SKFfiModel;

/// Resolve a model handle, rejecting null, freed and wrong-kind pointers.
pub(crate) fn sk_ffi_resolve_model(model: *mut SKFfiModel) -> Result<Arc<SKFfiModel>, SKFfiError> {
    let identifier = sk_ffi_handle_identifier(model);
    match sk_ffi_handle_lookup(identifier) {
        Ok(crate::handle::SKFfiResolvedHandle::Model(model)) => Ok(model),
        Ok(crate::handle::SKFfiResolvedHandle::Builder(_)) => Err(SKFfiError::InvalidHandle),
        Err(status) => Err(SKFfiError::from_status(status)),
    }
}
