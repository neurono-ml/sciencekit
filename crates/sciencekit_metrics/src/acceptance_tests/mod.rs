//! Acceptance tests for `sciencekit_metrics` (PRD §8.7 / §10.3).
//!
//! Covers tiny and large data, every averaging mode, the NaN-aware distance and
//! concurrent accumulation. Tolerances are documented per test.

mod averaging_modes_tests;
mod concurrency_tests;
mod distance_tests;
mod f32_precision_tests;
mod large_data_tests;
mod ranking_acceptance_tests;
mod small_data_tests;
