//! Classification metrics: label space, confusion matrix, accuracy and the
//! precision/recall/F1 family.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod accuracy;
mod averaging;
mod confusion;
mod label_space;
mod precision_recall_f1;

pub use accuracy::sk_accuracy_score;
pub use averaging::{SKAggregatedScores, SKAveraging, sk_aggregate_class_scores};
pub use confusion::{SKConfusionMatrix, sk_confusion_matrix};
pub use label_space::SKLabelSpace;
pub use precision_recall_f1::{SKClassScores, sk_per_class_precision_recall_f1};

#[cfg(test)]
mod accuracy_tests;
#[cfg(test)]
mod averaging_tests;
#[cfg(test)]
mod confusion_tests;
#[cfg(test)]
mod label_space_tests;
#[cfg(test)]
mod precision_recall_f1_tests;
