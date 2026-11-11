//! End-to-end acceptance tests for `sciencekit_decomposition` (PRD §8.7).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares test
//! submodules.

mod fixtures;

mod concurrency_and_metrics_tests;
mod large_and_solver_tests;
mod tiny_and_boundary_tests;
mod truncated_svd_acceptance_tests;
