//! Missing-data estimators (PRD §3.1 `sciencekit_impute`, wave `simple-imputers`).
//!
//! Three transformers: [`simple_imputer::SKSimpleImputer`], which fills the
//! missing cells of a numeric feature matrix with a per-column strategy
//! statistic, [`missing_indicator::SKMissingIndicator`], which reports where
//! those cells are, and [`knn::SKKNNImputer`], which fills a cell from the
//! observed values of the nearest training rows on the co-observed features.
//! All are builder-only, generic over the scalar `SKFloat`, record the
//! configured backend kind in every operation span; the first two are
//! sparse-aware through [`sprs`], while the nearest-neighbor imputer is the
//! documented dense-only exception (a CSR implicit zero is an observed zero).

pub mod knn;
pub mod missing_indicator;
pub mod missingness;
pub mod simple_imputer;

mod pipeline_adapter;

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod pipeline_adapter_tests;
#[cfg(test)]
mod sparse_streaming_test_datasets;
