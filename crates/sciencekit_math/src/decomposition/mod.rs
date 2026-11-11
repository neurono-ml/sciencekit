//! Decomposition kernels: truncated and randomized factorizations (PRD §4.4).
//!
//! The module hosts the deterministic randomized range finder, a truncated
//! singular-value decomposition over dense and sparse feature matrices.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

#[cfg(test)]
mod instrumentation;
mod centering;
mod kernel;
mod operator;
mod projection;
mod qr;
mod randomized;
mod result;
mod small_svd;

pub use projection::sk_seeded_projection;
pub use randomized::{SKRandomizedSvdBuilder, sk_randomized_svd};
pub use result::SKRandomizedSvdResult;

#[cfg(test)]
mod centered_tests;
#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
mod dense_tests;
#[cfg(test)]
mod projection_tests;
#[cfg(test)]
mod result_tests;
#[cfg(test)]
mod sparse_tests;
#[cfg(test)]
mod validation_tests;
