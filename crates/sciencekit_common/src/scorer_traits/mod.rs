//! Evaluation contracts (spec `scoring-contracts`).
//!
//! Supervised and unsupervised scorers with **dual input**: a pure form over
//! already-existing predictions/assignments (no re-inference) and a convenient
//! provided form that runs inference and delegates. Both are fallible. Scorers
//! are generic over the evaluated model so one scorer serves many algorithm
//! families. Defined here because they are part of the stable public contract;
//! concrete metrics arrive with the `sciencekit_metrics` crate.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod supervised;
mod unsupervised;

pub use supervised::SKSupervisedScorer;
pub use unsupervised::SKUnsupervisedScorer;

#[cfg(test)]
mod scorer_traits_tests;
