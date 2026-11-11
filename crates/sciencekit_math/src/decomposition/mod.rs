//! Decomposition kernels: truncated and randomized factorizations (PRD §4.4).
//!
//! The module hosts the deterministic randomized range finder, a truncated
//! singular-value decomposition over dense and sparse feature matrices.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod result;

pub use result::SKRandomizedSvdResult;

#[cfg(test)]
mod result_tests;
