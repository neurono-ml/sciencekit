//! Leaf-wise histogram tree growth and flat node storage (spec
//! `histogram-gradient-boosting`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod leaf_wise;
mod tree;

pub use leaf_wise::{
    SKHistogramGrowthParameters, SKHistogramGrowthResult, sk_grow_histogram_tree,
};
pub use tree::{SKHistogramNode, SKHistogramTree};

#[cfg(test)]
mod growth_tests;
