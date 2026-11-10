//! Record batch, table and chunked-array import into the neutral seam (spec
//! `arrow-columnar-interop`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod boolean_conversion;
mod chunked_conversion;
mod column_conversion;
mod dictionary_conversion;
mod float_conversion;
mod integer_conversion;
mod null_mapping;
mod record_batch_conversion;
mod text_conversion;

pub use chunked_conversion::sk_tabular_batch_from_column_chunks;
pub use column_conversion::sk_column_data_from_arrow_array;
pub use dictionary_conversion::{sk_dictionary_label_table, sk_label_tables_from_record_batch};
pub use record_batch_conversion::sk_tabular_batch_from_record_batch;
