//! `sciencekit_model_selection` — reproducible data partitioning for sciencekit
//! (spec `model-selection-splits`).
//!
//! The crate ships index-based train/test splitting, K-fold, stratified K-fold
//! and group K-fold. Splitters return [`SKIndexSplit`] values that address the
//! original rows, so no data is copied until a fold is materialized. Seeded
//! shuffling goes through the shared deterministic generator, so the same seed
//! and row count always produce the same partition.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod index_split;
pub mod split;

pub use index_split::SKIndexSplit;
pub use split::SKSplitter;

#[cfg(test)]
mod index_split_tests;
