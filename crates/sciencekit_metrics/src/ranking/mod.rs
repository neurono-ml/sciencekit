//! Ranking metrics: ROC/AUC, precision-recall/average precision, log loss,
//! balanced accuracy, Matthews correlation and the mergeable AUC accumulator
//! (spec `ranking-metrics`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod inputs;
mod roc;

pub use roc::{SKRocCurve, sk_roc_auc_score, sk_roc_curve};

#[cfg(test)]
mod roc_tests;
