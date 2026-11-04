//! The class-weighting policy shared by every linear classifier
//! (spec `stochastic-gradient-solver`, decision 11).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod expansion;
mod policy;

pub use expansion::sk_expand_class_weights;
pub use policy::SKClassWeight;

#[cfg(test)]
mod class_weight_tests;
