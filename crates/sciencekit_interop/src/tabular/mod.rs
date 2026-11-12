//! The tabular source contract shared by the Polars, DataFusion, Arrow and
//! Parquet adapters (spec `tabular-sources`).
//!
//! The neutral batch pipeline already converts engines into [`SKTabularBatch`]
//! values. This module adds the materialized counterpart a caller keeps after
//! conversion: a projected schema, a dense or sparse numeric feature block and
//! an owned categorical block, produced under one validated column
//! configuration.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod column_kind;
mod config;
mod dataset;
mod materialization;
mod nominal_matrix_owned;

pub use column_kind::SKColumnKind;
pub use config::{SKTabularSourceConfig, SKTabularSourceConfigBuilder};
pub use dataset::SKTabularDataset;
pub use nominal_matrix_owned::SKNominalMatrixOwned;

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod dataset_tests;
#[cfg(test)]
mod materialization_tests;
#[cfg(test)]
mod missing_tests;
