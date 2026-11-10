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

mod data_type_mapping;
mod import;
mod schema_mapping;
mod zero_copy;

pub use data_type_mapping::sk_logical_type_from_arrow_data_type;
pub use import::{
    SKArrowTextBorrowedView, sk_boolean_category_codes, sk_column_data_from_arrow_array,
    sk_dictionary_label_table, sk_label_tables_from_record_batch,
    sk_tabular_batch_from_column_chunks, sk_tabular_batch_from_record_batch, sk_text_borrowed_view,
};
pub use schema_mapping::{
    SK_ARROW_LOGICAL_TYPE_METADATA_KEY, SK_ARROW_ROLE_METADATA_KEY, SKArrowColumnRoles,
    sk_arrow_data_type_from_logical_type, sk_arrow_schema_from_tabular_schema,
    sk_tabular_schema_from_arrow_schema,
};
pub use zero_copy::{
    SKArrowBorrowedColumn, SKArrowConversionReport, sk_arrow_borrowed_column,
    sk_arrow_conversion_report,
};

#[cfg(test)]
mod boolean_import_tests;
#[cfg(test)]
mod chunked_import_tests;
#[cfg(test)]
mod dictionary_import_tests;
#[cfg(test)]
mod null_policy_tests;
#[cfg(test)]
mod record_batch_import_tests;
#[cfg(test)]
mod schema_mapping_tests;
#[cfg(test)]
mod wiring_tests;
#[cfg(test)]
mod zero_copy_tests;
