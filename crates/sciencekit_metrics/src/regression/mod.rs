//! Regression metrics: mean squared error, root mean squared error, mean
//! absolute error and the coefficient of determination.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod sums;

pub(crate) use sums::SKCompensatedSum;

#[cfg(test)]
mod sums_tests;
