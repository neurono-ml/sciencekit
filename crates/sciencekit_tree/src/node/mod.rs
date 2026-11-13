//! Flat-array tree structure (spec `decision-trees`, design decision 1).
//!
//! This is a **pure dispatcher** module: `mod.rs` declares submodules and
//! re-exports their public items; every implementation lives in its own file.

mod tree_node;
mod tree_structure;

pub use tree_node::SKTreeNode;
pub use tree_structure::SKTree;

#[cfg(test)]
mod node_tests;
