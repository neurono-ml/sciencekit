//! Typed column storage and scalar coercion (spec `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod coercion;
mod column_data;

pub use coercion::{sk_coerce_numeric_column, sk_integer_is_representable};
pub use column_data::SKColumnData;

#[cfg(test)]
mod coercion_tests;
#[cfg(test)]
mod column_data_tests;
