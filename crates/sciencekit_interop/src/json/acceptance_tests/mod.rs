//! End-to-end acceptance suite for the JSON debug format (PRD §8.7, spec
//! `json-debug-format`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares fixtures and
//! companion test modules; every scenario lives in its own file.

mod fixtures;

mod corruption_acceptance_tests;
mod state_acceptance_tests;
