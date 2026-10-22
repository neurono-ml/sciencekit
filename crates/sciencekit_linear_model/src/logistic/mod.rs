//! The shared logistic solver substrate for the linear classifier (spec
//! `logistic-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod class_weight;
pub mod config;
mod resolution;

pub use config::{SKClassWeight, SKLogisticPenalty, SKLogisticSolver, SKMultiClass};

#[cfg(test)]
mod config_tests;
