//! Runtime search machinery shared by the neighbor estimators
//! (spec `nearest-neighbor-estimators`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod algorithm;
mod brute_force;
mod configuration;
pub(crate) mod export_helpers;
mod index;
mod metric;
mod search;
mod training_data;
mod validation;
mod weights;

pub use algorithm::{
    SK_AUTO_BALL_TREE_MINIMUM_FEATURES, SK_AUTO_TREE_MINIMUM_TRAINING_ROWS,
    SKFeatureRepresentation, SKNeighborsAlgorithm, SKSearchStrategy, sk_resolve_search_strategy,
};
pub use brute_force::SKBruteForceSearcher;
pub(crate) use configuration::SKNeighborsConfiguration;
pub use metric::SKNeighborsMetric;
pub use training_data::SKTrainingData;
pub use weights::{SKNeighborsWeights, sk_weighted_average};

pub(crate) use brute_force::sk_brute_force_k_nearest;
pub(crate) use search::{SKFittedSearch, sk_prepare_search};
pub(crate) use validation::{sk_validate_query_features, sk_view_dimensions};

#[cfg(test)]
mod algorithm_tests;
#[cfg(test)]
mod weights_tests;
