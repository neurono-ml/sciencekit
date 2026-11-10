//! Ranking metrics: ROC/AUC, precision-recall/average precision, log loss,
//! balanced accuracy, Matthews correlation and the mergeable AUC accumulator
//! (spec `ranking-metrics`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod averaging;
mod balanced_accuracy;
mod inputs;
mod log_loss;
mod matthews;
mod multiclass_roc;
mod precision_recall;
mod roc;

pub use averaging::SKRankingAveraging;
pub use balanced_accuracy::sk_balanced_accuracy_score;
pub use log_loss::{sk_binary_log_loss, sk_log_loss};
pub use matthews::sk_matthews_correlation_coefficient;
pub use multiclass_roc::{sk_multiclass_roc_auc_score, sk_one_versus_rest_roc_auc_scores};
pub use precision_recall::{
    SKPrecisionRecallCurve, sk_average_precision_score, sk_precision_recall_curve,
};
pub use roc::{SKRocCurve, sk_roc_auc_score, sk_roc_curve};

#[cfg(test)]
mod balanced_mcc_tests;
#[cfg(test)]
mod log_loss_tests;
#[cfg(test)]
mod multiclass_roc_tests;
#[cfg(test)]
mod precision_recall_tests;
#[cfg(test)]
mod roc_tests;
