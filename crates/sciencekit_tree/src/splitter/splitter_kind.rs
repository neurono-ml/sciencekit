//! The split-selection strategy.

/// How each node chooses among its candidate thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKSplitter {
    /// Evaluate every candidate threshold and take the best (default).
    Best,
    /// Draw one random candidate threshold per considered feature.
    Random,
}
