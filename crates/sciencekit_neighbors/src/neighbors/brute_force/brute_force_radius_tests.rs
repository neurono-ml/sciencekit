//! Companion tests for the brute-force radius batch
//! (spec `nearest-neighbor-estimators`).

use ndarray::array;
use sciencekit_common::SKDataView;

use super::super::{SKBruteForceSearcher, SKNeighborsMetric, SKTrainingData};
use crate::searcher::SKNearestNeighborsSearcher;

#[test]
fn brute_force_radius_is_inclusive_and_ordered() {
    let training = array![[0.0_f64, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
    let queries = array![[0.0_f64, 0.0]];
    let searcher = SKBruteForceSearcher::build(
        SKTrainingData::Dense(training),
        SKNeighborsMetric::Euclidean,
    )
    .unwrap();

    let found = searcher
        .radius_neighbors(SKDataView::Dense(queries.view()), 2.0)
        .unwrap();

    assert_eq!(found.neighbors_count(0), 3);
    assert_eq!(found.indices(0), &[0, 1, 2]);
    assert_eq!(found.distances(0), &[0.0, 1.0, 2.0]);
}
