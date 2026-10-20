//! The `SKSimpleImputer` estimator (spec `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod configuration;
mod fitting;

pub use configuration::SKSimpleImputer;

pub(in crate::simple_imputer) use fitting::empty_columns;
