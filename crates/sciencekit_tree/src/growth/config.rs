//! Growth constraints and stochastic choices.

use crate::splitter::{SKMaximumFeatures, SKSplitter};

/// The constraints that bound tree growth.
#[derive(Debug, Clone)]
pub(crate) struct SKGrowthConfig<F> {
    /// Maximum depth; `None` grows until the other constraints stop it.
    pub(crate) maximum_depth: Option<usize>,
    /// A node must hold at least this many rows to be split.
    pub(crate) minimum_samples_split: usize,
    /// Each child must hold at least this many rows.
    pub(crate) minimum_samples_leaf: usize,
    /// How many features every split considers.
    pub(crate) maximum_features: SKMaximumFeatures<F>,
    /// Best or random threshold selection.
    pub(crate) splitter: SKSplitter,
    /// Seed of every stochastic choice.
    pub(crate) random_seed: u64,
}
