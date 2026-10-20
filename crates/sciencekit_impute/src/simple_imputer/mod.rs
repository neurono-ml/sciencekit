//! `SKSimpleImputer`: univariate missing-value imputation (spec `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod output;

pub use output::SKImputedOutput;

#[cfg(test)]
mod output_tests;
