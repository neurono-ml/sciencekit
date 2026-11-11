//! Kernel support-vector helpers and owned feature storage
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod distance;
mod resolved;

pub use distance::{sk_evaluate_kernel_diagonal, sk_evaluate_kernel_row};
pub use resolved::SKOwnedFeatureBlock;

#[cfg(test)]
mod kernel_tests;
