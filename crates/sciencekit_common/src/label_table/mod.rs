//! Canonical label tables (spec `data-view-boundary`).
//!
//! Deterministic canonicalization of nominal label sequences into compact
//! indices plus a reversible table — the foundation of classifier automatic
//! encoding (Phase 1+) and of explicit codecs.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod canonicalize;
// Private implementation module shadowing the parent name; allowed because the
// module stays private and every public item is re-exported below, so the
// same-name nesting never surfaces in public paths.
#[allow(clippy::module_inception)]
mod label_table;

pub use canonicalize::sk_canonicalize_labels;
pub use label_table::SKLabelTable;

#[cfg(test)]
mod label_table_tests;
