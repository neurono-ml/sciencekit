//! `sciencekit_ffi` — the stable C ABI for sciencekit (spec `ffi-abi`).
//!
//! The crate erases the Rust generic surface behind opaque handles, integer
//! status codes and array descriptors so Lua, Julia, Go and the JVM can call
//! the library without re-inventing unsafe glue. Python keeps its native PyO3
//! path; this ABI serves the other languages.
//!
//! # Memory ownership
//!
//! - Array **inputs** are borrowed for the duration of one call and never
//!   copied by the ABI; the caller owns the buffer.
//! - Array **outputs** are allocated by the library and must be released with
//!   [`sk_ffi_array_free`].
//! - Every object is an opaque handle created by a `*_create`/`sk_ffi_fit`
//!   function and released once with its matching `*_free` function.
//! - [`sk_ffi_last_error`] returns a thread-local string valid until the next
//!   ABI call on that thread; bindings copy it before returning to another
//!   thread.
//! - No Rust panic unwinds across the boundary: every entry point catches
//!   panics and reports [`SKFfiStatus::Internal`].
//!
//! **Naming:** public Rust items follow PRD §3.4; C symbols keep the stable
//! `sk_ffi_` prefix declared in `include/sciencekit.h`.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
// The C entry points take raw pointers by design. Their safety contract is
// documented per function (`# Safety`), validated before any dereference, and
// the functions are called from languages where a Rust `unsafe` marker has no
// meaning. Marking them `unsafe` would only burden the Rust-side tests.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod abi;
pub mod array;
pub mod boundary;
pub mod builder;
pub mod error;
pub mod error_kind;
pub mod handle;
pub mod model;
pub mod persistence;
pub mod status;

pub use abi::{
    SK_FFI_ABI_MAJOR, SK_FFI_ABI_MINOR, SK_FFI_ABI_PATCH, SK_FFI_FEATURE_F32, SK_FFI_FEATURE_F64,
    SK_FFI_FEATURE_LINEAR_REGRESSION, SK_FFI_FEATURE_MODEL_PERSISTENCE,
    SK_FFI_FEATURE_STANDARD_SCALER,
};
pub use array::{SKFfiArray, SKFfiDataType};
pub use boundary::{
    sk_ffi_array_free, sk_ffi_builder_create, sk_ffi_builder_free, sk_ffi_builder_set_backend,
    sk_ffi_builder_set_include_intercept, sk_ffi_builder_set_with_mean,
    sk_ffi_builder_set_with_std, sk_ffi_features, sk_ffi_fit, sk_ffi_internal_test_panic,
    sk_ffi_last_error, sk_ffi_model_free, sk_ffi_model_load, sk_ffi_model_save, sk_ffi_predict,
    sk_ffi_score, sk_ffi_status_name, sk_ffi_transform, sk_ffi_version,
};
pub use builder::SKFfiEstimatorFamily;
pub use error::{sk_ffi_clear_last_error, sk_ffi_set_last_error};
pub use status::SKFfiStatus;
