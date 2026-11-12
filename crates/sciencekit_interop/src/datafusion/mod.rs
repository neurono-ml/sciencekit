//! The DataFusion adapter (spec `tabular-sources`, feature `datafusion`).
//!
//! DataFusion `DataFrame` values convert into neutral batches, a streaming
//! source pulls record batches from a physical plan, and Parquet files are read
//! through a temporary session. DataFusion 46 carries its own Arrow 54, so the
//! adapter maps that vocabulary into the neutral one and never exposes an Arrow
//! type from this module.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod batch_conversion;
mod column_conversion;
mod dataframe_conversion;
mod dtype_mapping;
mod parquet_conversion;
mod stream_source;

pub use batch_conversion::sk_tabular_batch_from_datafusion_record_batch;
pub use column_conversion::sk_column_data_from_datafusion_array;
pub use dataframe_conversion::{
    sk_datafusion_runtime, sk_tabular_batches_from_dataframe,
    sk_tabular_dataset_from_dataframe, sk_tabular_dataset_from_dataframe_blocking,
};
pub use dtype_mapping::sk_logical_type_from_datafusion_data_type;
pub use parquet_conversion::{
    sk_dataframe_from_parquet, sk_tabular_dataset_from_parquet,
    sk_tabular_dataset_from_parquet_blocking,
};
pub use stream_source::SKDataFusionStreamSource;

#[cfg(test)]
mod datafusion_tests;
