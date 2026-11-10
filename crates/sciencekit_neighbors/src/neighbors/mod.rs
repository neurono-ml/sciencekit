//! Runtime search machinery shared by the neighbor estimators
//! (spec `nearest-neighbor-estimators`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod algorithm;
mod metric;
mod weights;

pub use algorithm::{
    SK_AUTO_BALL_TREE_MINIMUM_FEATURES, SK_AUTO_TREE_MINIMUM_TRAINING_ROWS,
    SKFeatureRepresentation, SKNeighborsAlgorithm, SKSearchStrategy,
    sk_resolve_search_strategy,
};
pub use metric::SKNeighborsMetric;
pub use weights::SKNeighborsWeights;

#[cfg(test)]
mod algorithm_tests;
#[cfg(test)]
mod weights_tests;
