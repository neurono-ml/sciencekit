//! The neutral schema module (spec `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod column_role;
mod column_schema;
mod logical_type;
mod tabular_schema;

pub use column_role::SKColumnRole;
pub use column_schema::SKColumnSchema;
pub use logical_type::SKLogicalType;
pub use tabular_schema::SKTabularSchema;

#[cfg(test)]
mod schema_tests;
#[cfg(test)]
mod tabular_schema_tests;
