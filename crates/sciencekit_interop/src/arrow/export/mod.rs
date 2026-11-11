//! Export of sciencekit values to Arrow (spec `arrow-columnar-interop`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod array_export;
mod csr_export;
mod dense_export;
mod record_batch_export;

pub use array_export::{SKArrowExportedArray, sk_export_prediction};
pub use csr_export::{
    SK_ARROW_NUMBER_OF_COLUMNS_METADATA_KEY, SK_ARROW_NUMBER_OF_ROWS_METADATA_KEY,
    SKArrowCsrExport, sk_export_csr_matrix,
};
pub use dense_export::sk_export_dense_matrix;
pub use record_batch_export::{
    SKArrowBatchExport, SKArrowExportedRecordBatch, sk_export_data_batch,
    sk_export_supervised_batch,
};

#[cfg(test)]
mod export_tests;
