//! Split search (spec `decision-trees`).

mod best;
mod candidate;
mod scan;

pub(crate) use best::sk_best_split;
pub use candidate::SKSplitCandidate;
pub(crate) use scan::{SKScanContext, SKThresholdChoice, sk_scan_feature};

/// The feature-subsampling policy.
#[derive(Debug, Clone, PartialEq)]
pub enum SKMaximumFeatures<F> {
    /// Consider every feature (default).
    All,
    /// Consider exactly this many features.
    Count(usize),
    /// Consider this fraction of the features.
    Fraction(F),
    /// Consider the square root of the feature count.
    Sqrt,
    /// Consider the base-2 logarithm of the feature count.
    Log2,
}

/// The split-selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKSplitter {
    /// Evaluate every candidate threshold and take the best (default).
    Best,
    /// Draw one random candidate threshold per considered feature.
    Random,
}

#[cfg(test)]
mod splitter_best_tests;
#[cfg(test)]
mod splitter_dense_tests;
#[cfg(test)]
mod splitter_random_tests;
#[cfg(test)]
mod splitter_sparse_tests;
