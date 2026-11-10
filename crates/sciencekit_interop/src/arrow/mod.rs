//! The Apache Arrow adapter (spec `arrow-columnar-interop`, feature `arrow`).
//!
//! Arrow is the first concrete engine behind the neutral seam: record batches,
//! tables and batch readers become [`crate::batch::SKTabularBatch`] values, and
//! sciencekit results go back out as Arrow arrays, record batches and C Data
//! Interface structures. The module is compiled only when the `arrow` feature
//! is enabled, so the neutral core stays free of Arrow dependencies.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

#[cfg(test)]
mod wiring_tests;
