//! Acceptance tests for the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).
//!
//! The PRD acceptance contract: the estimator runs with lots and little data,
//! under concurrency, fills every missing pattern and produces metrics.
//!
//! Pure dispatcher module: declares its submodules; the fixtures and the
//! scenarios live in their own files under this folder.

mod fixtures;

mod concurrency_tests;

mod indicator_composition_tests;

mod large_data_tests;

mod metric_tests;

mod missing_pattern_tests;

mod tiny_data_tests;

mod weighting_tests;
