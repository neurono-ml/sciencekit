//! The neutral schema module (spec `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod column_role;
mod column_schema;
mod logical_type;

pub use column_role::SKColumnRole;
pub use column_schema::SKColumnSchema;
pub use logical_type::SKLogicalType;

#[cfg(test)]
mod schema_tests;
