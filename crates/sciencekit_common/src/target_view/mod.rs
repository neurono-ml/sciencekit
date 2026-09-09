//! Target boundary: continuous/integer/nominal target representation
//! (spec `data-view-boundary`).
//!
//! Storage ≠ interpretation: `[1,2,3]` is continuous for a regressor and
//! categorical for a classifier. The view describes *how* data is stored.
//! Continuous targets use `f64` independently of the feature dtype `F`
//! (design decision 4). Integer targets elevate losslessly to continuous;
//! nominal targets borrow their text.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod conversions;
// Private implementation module shadowing the parent name; allowed because the
// module stays private and every public item is re-exported below, so the
// same-name nesting never surfaces in public paths.
#[allow(clippy::module_inception)]
mod target_view;

pub use target_view::SKTargetView;

#[cfg(test)]
mod target_view_tests;
