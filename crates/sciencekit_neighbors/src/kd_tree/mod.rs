//! KD-tree construction and queries.
//!
//! This is a pure dispatcher module: implementation and companion tests live in
//! their own files.

mod build;
mod builder;
mod node;

pub use build::SKKDTree;
pub use builder::SKKDTreeBuilder;

#[cfg(test)]
mod build_tests;
