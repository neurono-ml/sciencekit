//! The documented missing-code sentinel for integer-coded categorical features.

/// The sentinel code marking a missing categorical entry in integer-coded
/// columns.
///
/// It is never registered as a category and always encodes to an all-zero
/// column block, so every consumer (encoders, imputers) agrees on missing
/// handling. `-1` matches scikit-learn's convention.
pub const SK_CATEGORICAL_MISSING_CODE: i64 = -1;
