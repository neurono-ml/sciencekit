//! The missing-value marker rules (spec `simple-imputer`).
//!
//! One rule per representation. Dense: a cell is missing exactly when it
//! equals the configured marker, with `NaN` compared by `is_nan()` since
//! `NaN == NaN` is always false — so the default `NaN` marker recognizes
//! `NaN` cells and nothing else. Sparse: explicit entries follow the same
//! rule, and the implicit zeros join the missing set only when the marker
//! is `0.0`, the standard "my dense matrix lost its shape" idiom; any other
//! marker leaves the structural zeros observed.

use sciencekit_common::SKFloat;

/// Whether a dense (or explicit sparse) cell counts as missing under `marker`.
///
/// A `NaN` marker matches `NaN` cells through `is_nan()` (since `NaN == NaN`
/// is always false); any other marker matches by equality, so a `NaN` cell is
/// never missing under a non-`NaN` marker.
pub fn sk_is_missing<F: SKFloat>(value: F, marker: F) -> bool {
    if marker.is_nan() {
        value.is_nan()
    } else {
        value == marker
    }
}

/// Whether a sparse matrix's implicit zeros count as missing under `marker`.
///
/// Only the `0.0` marker claims the implicit zeros; every other marker,
/// including the default `NaN`, leaves the structural zeros observed.
pub fn sk_sparse_marks_implicit_zeros<F: SKFloat>(marker: F) -> bool {
    marker == F::zero()
}
