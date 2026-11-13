//! Advanced structure-only tree growth and leaf-value replacement for gradient
//! boosting (spec `decision-trees`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items.

mod build;
mod constraints;
mod criterion;
mod leaf_values;

pub use build::SKTreeStructure;
pub use constraints::{SKStructureConstraints, SKStructureConstraintsBuilder};

#[cfg(test)]
mod structure_tests;
