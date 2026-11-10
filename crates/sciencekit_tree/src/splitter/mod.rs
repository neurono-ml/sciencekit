//! Split search (spec `decision-trees`).

mod best;
mod candidate;
mod scan;

pub(crate) use best::sk_best_split;
pub use candidate::SKSplitCandidate;
pub(crate) use scan::{SKScanContext, sk_classification_impurity};
#[cfg(test)]
pub(crate) use scan::{SKThresholdChoice, sk_scan_feature};

mod maximum_features;
mod splitter_kind;

pub use maximum_features::SKMaximumFeatures;
pub use splitter_kind::SKSplitter;

#[cfg(test)]
mod splitter_best_tests;
#[cfg(test)]
mod splitter_dense_tests;
#[cfg(test)]
mod splitter_random_tests;
#[cfg(test)]
mod splitter_sparse_tests;
