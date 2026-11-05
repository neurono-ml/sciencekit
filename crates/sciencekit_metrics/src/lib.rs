//! `sciencekit_metrics` — evaluation metrics for sciencekit (PRD §12, Phase 6).
//!
//! This crate provides the concrete metrics behind the shared scoring contracts:
//! whole-input functions, scorer adapters usable by any compatible fitted model,
//! and mergeable streaming accumulators.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![warn(missing_docs)]

pub mod accumulators;
pub mod classification;
pub mod regression;
pub mod scorers;

mod validation;

#[cfg(test)]
mod validation_tests;
