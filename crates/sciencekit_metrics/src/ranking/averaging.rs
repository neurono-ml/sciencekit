//! Aggregation policies for the multiclass ranking metrics.

/// How per-class ranking scores are aggregated into one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKRankingAveraging {
    /// The unweighted mean over every class.
    Macro,
    /// One rank-sum score over every (observation, class) pair.
    Micro,
    /// The mean weighted by each class's true support.
    Weighted,
}
