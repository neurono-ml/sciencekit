//! The missing-value policy module (spec `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod missing_value_policy;

pub use missing_value_policy::SKMissingValuePolicy;

#[cfg(test)]
mod missing_value_policy_tests;
