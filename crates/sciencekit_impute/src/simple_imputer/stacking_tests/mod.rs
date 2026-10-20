//! Companion tests for the `add_indicator` stacking rule (spec
//! `simple-imputer`).
//!
//! The stacked output is the imputed features followed, column by column, by
//! the missingness mask of the features that carried a missing cell at fit
//! time. The scope keeps growing through `partial_fit` and through the
//! streaming driver.
//!
//! Pure dispatcher module: declares its submodules; the fixtures and the
//! scenarios live in their own files under this folder.

mod fixtures;

mod dense_tests;

mod sparse_tests;
