//! The fitted indicator model (spec `missing-indicator`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod core_implementation;
pub(crate) mod export;
mod transform;

pub use core_implementation::SKMissingIndicatorModel;
