//! End-to-end acceptance tests for the extended Safetensors format (PRD §8.7,
//! spec `safetensors-format`).
//!
//! Pure dispatcher module for companion acceptance suites.

mod concurrency_and_corruption_tests;
mod fixtures;
mod state_acceptance_tests;
