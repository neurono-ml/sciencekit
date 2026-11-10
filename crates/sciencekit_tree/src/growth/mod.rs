//! Shared recursive growth engine for both estimators.
//!
//! This is a **pure dispatcher** module: `mod.rs` declares the submodules and
//! re-exports their items; the implementation lives in `engine`.

mod config;
mod criterion;
mod engine;
mod feature_selection;

pub(crate) use config::SKGrowthConfig;
pub(crate) use criterion::SKTreeCriterion;
pub(crate) use engine::sk_grow_tree;

#[cfg(test)]
mod feature_subsampling_tests;
#[cfg(test)]
pub(crate) use feature_selection::sk_select_features;

#[cfg(test)]
mod growth_tests;
