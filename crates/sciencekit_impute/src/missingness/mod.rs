//! Missing-value marker semantics shared by the impute estimators (spec
//! `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod marker;

pub use marker::{sk_is_missing, sk_sparse_marks_implicit_zeros};

#[cfg(test)]
mod marker_tests;
