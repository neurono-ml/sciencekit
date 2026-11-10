//! Export of sciencekit values to Arrow (spec `arrow-columnar-interop`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod dense_export;

pub use dense_export::sk_export_dense_matrix;
