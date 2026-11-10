//! Candidate/fold evaluation, score aggregation and best selection
//! (spec `grid-search`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod candidate_evaluation;
mod fold_scorer;

pub(crate) use candidate_evaluation::sk_evaluate_candidates;
pub(crate) use fold_scorer::SKFoldScorer;

#[cfg(test)]
mod evaluation_tests;
#[cfg(test)]
mod selection_tests;