//! Pure dispatcher: the opaque handle registry. Carries no implementation
//! logic.

mod registry;

pub(crate) use registry::{
    SKFfiHandle, SKFfiResolvedHandle, sk_ffi_handle_insert, sk_ffi_handle_lookup,
    sk_ffi_handle_remove,
};
