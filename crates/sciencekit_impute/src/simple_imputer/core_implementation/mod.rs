//! The `SKSimpleImputer` estimator (spec `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod configuration;
mod fitting;
mod validation;

pub use configuration::SKSimpleImputer;

pub(in crate::simple_imputer) use validation::empty_columns;
