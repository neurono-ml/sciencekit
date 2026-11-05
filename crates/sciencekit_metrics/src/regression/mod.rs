//! Regression metrics: mean squared error, root mean squared error, mean
//! absolute error and the coefficient of determination.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod inputs;
mod mean_errors;
mod sums;

pub(crate) use sums::SKCompensatedSum;

pub use mean_errors::{
    sk_mean_absolute_error, sk_mean_squared_error, sk_root_mean_squared_error,
};

#[cfg(test)]
mod mean_errors_tests;
#[cfg(test)]
mod sums_tests;
