//! The `ffi` family: the stable C ABI.
//!
//! Every item here resolves to `sciencekit_ffi`. Rust callers usually do not
//! need this module; language bindings do. The `sciencekit.h` header is the
//! authoritative C-side contract.

pub use sciencekit_ffi::{
    SK_FFI_ABI_MAJOR, SK_FFI_ABI_MINOR, SK_FFI_ABI_PATCH, SK_FFI_FEATURE_F32, SK_FFI_FEATURE_F64,
    SK_FFI_FEATURE_LINEAR_REGRESSION, SK_FFI_FEATURE_MODEL_PERSISTENCE,
    SK_FFI_FEATURE_STANDARD_SCALER,
};
pub use sciencekit_ffi::{SKFfiArray, SKFfiDataType, SKFfiEstimatorFamily, SKFfiStatus};
pub use sciencekit_ffi::{
    abi, array, boundary, builder, error, error_kind, handle, model, persistence, status,
};
pub use sciencekit_ffi::{
    sk_ffi_array_free, sk_ffi_builder_create, sk_ffi_builder_free, sk_ffi_builder_set_backend,
    sk_ffi_builder_set_include_intercept, sk_ffi_builder_set_with_mean,
    sk_ffi_builder_set_with_std, sk_ffi_clear_last_error, sk_ffi_features, sk_ffi_fit,
    sk_ffi_last_error, sk_ffi_model_free, sk_ffi_model_load, sk_ffi_model_save, sk_ffi_predict,
    sk_ffi_score, sk_ffi_set_last_error, sk_ffi_status_name, sk_ffi_transform, sk_ffi_version,
};
