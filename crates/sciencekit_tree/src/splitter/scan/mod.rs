//! Per-feature split scans for every supported criterion.
//!
//! This is a **pure dispatcher** module: `mod.rs` declares the submodules and
//! re-exports their public items; the scans and the dispatch live in their own
//! files.

mod absolute;
mod classification;
mod dispatch;
mod ranked;
mod selection;
mod squared;

pub(crate) use classification::sk_classification_impurity;
pub(crate) use dispatch::{SKScanContext, SKThresholdChoice, sk_scan_feature};
pub(crate) use selection::{
    sk_better_candidate, sk_choose_boundary, sk_midpoint, sk_summary_for_indices,
};
