//! The ABI version, feature flags and the checked entry-point inventory.
//!
//! The header consistency test reads this inventory so the hand-maintained
//! `include/sciencekit.h` cannot silently drift away from the exported
//! symbols. Additive changes bump the minor component; a breaking change
//! bumps the major one.

/// Major ABI version. A binding refuses a library with a different major.
pub const SK_FFI_ABI_MAJOR: i32 = 1;

/// Minor ABI version. Additive symbols bump this component.
pub const SK_FFI_ABI_MINOR: i32 = 0;

/// Patch ABI version. Implementation fixes keep the ABI stable.
pub const SK_FFI_ABI_PATCH: i32 = 0;

/// Feature bit: the linear-regression family is available.
pub const SK_FFI_FEATURE_LINEAR_REGRESSION: u64 = 1 << 0;

/// Feature bit: the standard-scaler family is available.
pub const SK_FFI_FEATURE_STANDARD_SCALER: u64 = 1 << 1;

/// Feature bit: models can be saved to and loaded from the interop state
/// container.
pub const SK_FFI_FEATURE_MODEL_PERSISTENCE: u64 = 1 << 2;

/// Feature bit: single-precision descriptors are accepted.
pub const SK_FFI_FEATURE_F32: u64 = 1 << 3;

/// Feature bit: double-precision descriptors are accepted.
pub const SK_FFI_FEATURE_F64: u64 = 1 << 4;

/// The feature bitmask the loaded library reports.
pub const SK_FFI_FEATURES: u64 = SK_FFI_FEATURE_LINEAR_REGRESSION
    | SK_FFI_FEATURE_STANDARD_SCALER
    | SK_FFI_FEATURE_MODEL_PERSISTENCE
    | SK_FFI_FEATURE_F32
    | SK_FFI_FEATURE_F64;

/// Every exported C symbol, in header order.
pub const SK_FFI_ENTRY_POINTS: &[&str] = &[
    "sk_ffi_version",
    "sk_ffi_features",
    "sk_ffi_status_name",
    "sk_ffi_last_error",
    "sk_ffi_builder_create",
    "sk_ffi_builder_set_include_intercept",
    "sk_ffi_builder_set_with_mean",
    "sk_ffi_builder_set_with_std",
    "sk_ffi_builder_set_backend",
    "sk_ffi_builder_free",
    "sk_ffi_fit",
    "sk_ffi_model_free",
    "sk_ffi_predict",
    "sk_ffi_transform",
    "sk_ffi_score",
    "sk_ffi_model_save",
    "sk_ffi_model_load",
    "sk_ffi_array_free",
];

#[cfg(test)]
#[path = "abi_tests.rs"]
mod abi_tests;
