//! Classification metrics: label space, confusion matrix, accuracy and the
//! precision/recall/F1 family.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod accuracy;
mod confusion;
mod label_space;

pub use accuracy::sk_accuracy_score;
pub use confusion::{SKConfusionMatrix, sk_confusion_matrix};
pub use label_space::SKLabelSpace;

#[cfg(test)]
mod accuracy_tests;
#[cfg(test)]
mod confusion_tests;
#[cfg(test)]
mod label_space_tests;
