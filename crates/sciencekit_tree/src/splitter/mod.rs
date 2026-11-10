//! Split search (spec `decision-trees`).

mod candidate;

pub use candidate::SKSplitCandidate;

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
