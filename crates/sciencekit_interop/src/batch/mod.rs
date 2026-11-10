//! Column-major batches and their conversion into the library's views (spec
//! `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod tabular_batch;

pub use tabular_batch::SKTabularBatch;

#[cfg(test)]
mod tabular_batch_tests;
