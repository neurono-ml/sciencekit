//! Acceptance suite for the linear support-vector classifier (PRD §8.7,
//! spec `linear-support-vector-classifier`).
//!
//! Pure dispatcher module: every scenario lives in its own companion file.

mod large_sparse_tests;
mod metrics_tests;
mod parallel_equals_sequential_tests;
mod tiny_data_tests;
