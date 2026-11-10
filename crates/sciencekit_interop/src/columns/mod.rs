//! Typed column storage and scalar coercion (spec `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod column_data;

pub use column_data::SKColumnData;

#[cfg(test)]
mod column_data_tests;
