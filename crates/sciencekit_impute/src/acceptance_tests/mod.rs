//! Acceptance tests for the impute crate (specs `simple-imputer` and
//! `missing-indicator`).
//!
//! The PRD acceptance contract: every estimator runs with lots and little data,
//! under concurrency, and the fitted state stays reachable as the export
//! boundary. Dense and sparse inputs must answer alike.
//!
//! Pure dispatcher module: declares its submodules; the fixtures and the
//! scenarios live in their own files under this folder.

mod fixtures;

mod concurrency_tests;

mod large_data_tests;

mod representation_tests;

mod small_data_tests;
