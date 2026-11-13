//! The `neighbors` family: exact spatial indexes and nearest-neighbor models.
//!
//! Every item here resolves to `sciencekit_neighbors`.

pub use sciencekit_neighbors::ball_tree::{SKBallTree, SKBallTreeBuilder};
pub use sciencekit_neighbors::classifier::{
    SKKNeighborsClassifier, SKKNeighborsClassifierBuilder, SKKNeighborsClassifierModel,
};
pub use sciencekit_neighbors::kd_tree::{SKKDTree, SKKDTreeBuilder};
pub use sciencekit_neighbors::metric::{SKNeighborMetric, SKRequestedNeighborMetric};
pub use sciencekit_neighbors::neighbors::{
    SKBruteForceSearcher, SKFeatureRepresentation, SKNeighborsAlgorithm, SKNeighborsMetric,
    SKNeighborsWeights, SKSearchStrategy, SKTrainingData, sk_resolve_search_strategy,
    sk_weighted_average,
};
pub use sciencekit_neighbors::regressor::{
    SKKNeighborsRegressor, SKKNeighborsRegressorBuilder, SKKNeighborsRegressorModel,
};
pub use sciencekit_neighbors::searcher::{
    SKNearestNeighborsSearcher, SKNeighborQuery, SKRadiusNeighbors,
};
pub use sciencekit_neighbors::{
    ball_tree, classifier, kd_tree, metric, neighbors, regressor, searcher,
};
