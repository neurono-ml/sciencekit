//! Per-column fit statistics (spec `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod dense;
mod observed_column;
mod sparse;

pub(super) use dense::dense_column_statistics;
pub(super) use sparse::sparse_column_statistics;

#[cfg(test)]
mod observed_column_tests;
