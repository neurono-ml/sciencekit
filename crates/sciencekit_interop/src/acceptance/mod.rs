//! End-to-end acceptance tests for the interop seam (PRD §8.7/§10.3, spec
//! `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module for the acceptance suite: it declares
//! and re-exports the fixtures and the scenario modules.

pub(crate) mod fixtures;

#[cfg(test)]
mod conversion_tests;
#[cfg(test)]
mod error_tests;
