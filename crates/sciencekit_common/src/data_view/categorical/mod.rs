//! Categorical feature boundary: the categorical counterpart of `SKDataView`.
//!
//! Categorical features — borrowed text labels or small integer codes — are not
//! numeric feature vectors, so they get their own zero-copy, extensible view
//! entered through the same fallible conversion seam (spec
//! `data-view-boundary`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

#[allow(clippy::module_inception)]
mod categorical_view;
mod conversions;
mod missing;
mod nominal_matrix;

pub use categorical_view::SKCategoricalView;
pub use missing::SK_CATEGORICAL_MISSING_CODE;
pub use nominal_matrix::SKNominalMatrixView;

#[cfg(test)]
mod categorical_tests;
