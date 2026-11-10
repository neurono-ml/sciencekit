//! Candidate/fold evaluation, score aggregation and best selection
//! (spec `grid-search`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod fold_scorer;

pub(crate) use fold_scorer::SKFoldScorer;
