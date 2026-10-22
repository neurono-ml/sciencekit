//! The shared logistic solver substrate for the linear classifier (spec
//! `logistic-solver`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod accumulator;
mod class_weight;
pub mod config;
mod design;
pub(crate) mod objective;
pub(crate) mod resolution;
pub(crate) mod solver;
mod targets;

pub(crate) use accumulator::SKLogisticState;
pub use config::{SKClassWeight, SKLogisticPenalty, SKLogisticSolver, SKMultiClass};
pub(crate) use solver::SKLogisticSolution;
pub(crate) use targets::{sk_canonicalize_against, sk_canonicalize_targets};

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod design_tests;
