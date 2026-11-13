//! The Polars adapter (spec `tabular-sources`, feature `polars`).
//!
//! Polars `DataFrame` and `LazyFrame` values convert into the neutral tabular
//! batch and the materialized [`crate::tabular::SKTabularDataset`]. The module
//! is compiled only when the `polars` feature is enabled, so the default build
//! stays free of the engine.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod column_conversion;
mod dataframe_conversion;
mod dtype_mapping;
mod lazy_source;

pub use column_conversion::sk_column_data_from_polars_series;
pub use dataframe_conversion::{
    sk_tabular_batch_from_dataframe, sk_tabular_dataset_from_dataframe,
};
pub use dtype_mapping::sk_logical_type_from_polars_data_type;
pub use lazy_source::SKPolarsLazySource;

#[cfg(test)]
mod polars_tests;
