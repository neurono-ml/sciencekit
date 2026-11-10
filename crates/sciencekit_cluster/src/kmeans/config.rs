//! KMeans configuration vocabulary (spec `k-means`).

/// Centroid initialization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKKMeansInitialization {
    /// D² sampling: the first center is uniform, each next center is drawn with
    /// probability proportional to its squared distance to the nearest chosen
    /// center.
    KMeansPlusPlus,
    /// Uniformly drawn distinct training rows.
    Random,
}

impl Default for SKKMeansInitialization {
    fn default() -> Self {
        Self::KMeansPlusPlus
    }
}

/// Assignment algorithm requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKKMeansAlgorithm {
    /// Elkan for dense input, Lloyd for sparse input.
    Auto,
    /// Plain Lloyd iteration.
    Lloyd,
    /// Triangle-inequality Elkan acceleration (dense input only).
    Elkan,
}

impl Default for SKKMeansAlgorithm {
    fn default() -> Self {
        Self::Auto
    }
}
