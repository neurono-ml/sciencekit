//! The C ABI boundary: every `extern "C"` entry point and the only module
//! allowed to dereference caller pointers. Carries no logic beyond argument
//! validation and dispatch.

mod array_entry_points;
mod builder_entry_points;
mod guard;
mod misc_entry_points;
mod model_entry_points;
mod pointers;

pub use array_entry_points::sk_ffi_array_free;
pub use builder_entry_points::{
    sk_ffi_builder_create, sk_ffi_builder_free, sk_ffi_builder_set_backend,
    sk_ffi_builder_set_include_intercept, sk_ffi_builder_set_with_mean,
    sk_ffi_builder_set_with_std,
};
#[cfg(any(test, feature = "test-hook"))]
pub use misc_entry_points::sk_ffi_internal_test_panic;
pub use misc_entry_points::{
    sk_ffi_features, sk_ffi_last_error, sk_ffi_status_name, sk_ffi_version,
};
pub use model_entry_points::{
    sk_ffi_fit, sk_ffi_model_free, sk_ffi_model_load, sk_ffi_model_save, sk_ffi_predict,
    sk_ffi_score, sk_ffi_transform,
};
