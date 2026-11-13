//! The explicit column selection module (spec `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod column_reference;
mod column_selection;

pub use column_reference::SKColumnReference;
pub use column_selection::{SKColumnSelection, SKColumnSelectionBuilder, SKResolvedSelection};

#[cfg(test)]
mod selection_tests;
