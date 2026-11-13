//! Companion validation tests for the brute-force searcher
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};

use super::super::{SKBruteForceSearcher, SKNeighborsMetric, SKTrainingData};
use crate::searcher::SKNearestNeighborsSearcher;

fn dense_training() -> Array2<f64> {
    Array2::from_shape_fn((40, 3), |(row, column)| {
        ((row * 7 + column * 13) % 97) as f64 / 3.0 + row as f64 * 0.001
    })
}

fn dense_searcher() -> SKBruteForceSearcher<f64> {
    SKBruteForceSearcher::build(
        SKTrainingData::Dense(dense_training()),
        SKNeighborsMetric::Euclidean,
    )
    .expect("the dense fixture is valid")
}

#[test]
fn brute_force_rejects_a_feature_count_mismatch() {
    let searcher = dense_searcher();
    let queries = array![[0.0_f64, 0.0]];

    match searcher.k_nearest(SKDataView::Dense(queries.view()), 1) {
        Err(SKError::ShapeMismatch { expected, found }) => {
            assert_eq!(expected, vec![1, 3]);
            assert_eq!(found, vec![1, 2]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn brute_force_rejects_a_non_finite_query() {
    let searcher = dense_searcher();
    let queries = array![[0.0_f64, f64::NAN, 0.0]];

    match searcher.k_nearest(SKDataView::Dense(queries.view()), 1) {
        Err(SKError::NonFiniteValue { position }) => assert_eq!(position, 1),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn brute_force_rejects_an_excessive_neighbor_count() {
    let searcher = dense_searcher();

    match searcher.k_nearest(SKDataView::Dense(dense_training().view()), 41) {
        Err(SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "nearest_neighbors_count");
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}
