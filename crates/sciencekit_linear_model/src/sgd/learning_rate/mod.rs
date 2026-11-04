//! The learning-rate schedules of the stochastic-gradient substrate
//! (spec `stochastic-gradient-solver`, decision 4).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod policy;

pub use policy::SKSGDLearningRate;
