//! Ranking metrics: ROC/AUC, precision-recall/average precision, log loss,
//! balanced accuracy, Matthews correlation and the mergeable AUC accumulator
//! (spec `ranking-metrics`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod averaging;
mod inputs;
mod multiclass_roc;
mod roc;

pub use averaging::SKRankingAveraging;
pub use multiclass_roc::{sk_multiclass_roc_auc_score, sk_one_versus_rest_roc_auc_scores};
pub use roc::{SKRocCurve, sk_roc_auc_score, sk_roc_curve};

#[cfg(test)]
mod multiclass_roc_tests;
#[cfg(test)]
mod roc_tests;
