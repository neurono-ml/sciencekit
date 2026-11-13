//! Dataset, model and prediction adapters.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod dataset;
mod model;
mod predictions;
#[cfg(feature = "parquet")]
mod predictions_parquet;

#[allow(unused_imports)]
pub use dataset::{SKDataset, SKDatasetFormat, sk_load_dataset};
#[allow(unused_imports)]
pub use model::{SKModelFormat, sk_inspect, sk_load_native, sk_load_state, sk_save_state};
#[allow(unused_imports)]
pub use predictions::{SKOutputFormat, sk_write_predictions, sk_write_transformed_matrix};

#[cfg(test)]
mod dataset_tests;
#[cfg(test)]
mod model_tests;
