//! Classification metrics: label space, confusion matrix, accuracy and the
//! precision/recall/F1 family.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod label_space;

pub use label_space::SKLabelSpace;

#[cfg(test)]
mod label_space_tests;
