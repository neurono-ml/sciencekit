//! Companion tests for the brute-force searcher
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};
use sprs::{CsMat, TriMat};

use super::super::{SKBruteForceSearcher, SKNeighborsMetric, SKTrainingData};
use super::sk_brute_force_k_nearest_with_budget;
use crate::kd_tree::SKKDTreeBuilder;
use crate::searcher::SKNearestNeighborsSearcher;

fn dense_training() -> Array2<f64> {
    // Every row is distinct, so the brute-force and tree kernels cannot
    // disagree on an exactly tied distance caused by duplicate rows.
    Array2::from_shape_fn((40, 3), |(row, column)| {
        ((row * 7 + column * 13) % 97) as f64 / 3.0 + row as f64 * 0.001
    })
}

fn sparse_example() -> CsMat<f64> {
    let mut triplet = TriMat::new((3, 3));
    triplet.add_triplet(0, 0, 1.0);
    triplet.add_triplet(1, 1, 2.0);
    triplet.add_triplet(2, 0, 3.0);
    triplet.to_csr()
}

fn dense_searcher() -> SKBruteForceSearcher<f64> {
    SKBruteForceSearcher::build(
        SKTrainingData::Dense(dense_training()),
        SKNeighborsMetric::Euclidean,
    )
    .expect("the dense fixture is valid")
}

#[test]
fn brute_force_agrees_with_the_kd_tree_on_dense_data() {
    let training = dense_training();
    let queries = Array2::from_shape_fn((9, 3), |(row, column)| {
        ((row * 5 + column * 3) % 7) as f64 / 2.0
    });
    let tree = SKKDTreeBuilder::new()
        .leaf_size(4)
        .build(SKDataView::Dense(training.view()))
        .expect("the tree builds");
    let expected = tree
        .k_nearest(SKDataView::Dense(queries.view()), 5)
        .expect("the tree query works");

    let searcher = SKBruteForceSearcher::build(
        SKTrainingData::Dense(training),
        SKNeighborsMetric::Euclidean,
    )
    .unwrap();
    let found = searcher
        .k_nearest(SKDataView::Dense(queries.view()), 5)
        .unwrap();

    assert_eq!(found.indices(), expected.indices());
    for (found, expected) in found.distances().iter().zip(expected.distances()) {
        assert!((found - expected).abs() < 1e-12);
    }
}

#[test]
fn brute_force_hand_computed_dense_example() {
    let training = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let queries = array![[0.5_f64, 0.5]];
    let searcher = SKBruteForceSearcher::build(
        SKTrainingData::Dense(training),
        SKNeighborsMetric::Euclidean,
    )
    .unwrap();

    let found = searcher
        .k_nearest(SKDataView::Dense(queries.view()), 2)
        .unwrap();

    assert_eq!(found.indices().row(0).to_vec(), vec![0, 1]);
    assert!((found.distances()[[0, 0]] - 0.5_f64.sqrt()).abs() < 1e-12);
    assert!((found.distances()[[0, 1]] - 0.5_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn brute_force_hand_computed_sparse_example() {
    let training = sparse_example();
    let mut query = TriMat::new((1, 3));
    query.add_triplet(0, 0, 1.0);
    let query = query.to_csr();
    let searcher = SKBruteForceSearcher::build(
        SKTrainingData::Sparse(training),
        SKNeighborsMetric::Euclidean,
    )
    .unwrap();

    let found = searcher
        .k_nearest(SKDataView::Sparse(query.view()), 3)
        .unwrap();

    assert_eq!(found.indices().row(0).to_vec(), vec![0, 2, 1]);
    assert_eq!(found.distances()[[0, 0]], 0.0);
    assert!((found.distances()[[0, 1]] - 2.0).abs() < 1e-12);
    assert!((found.distances()[[0, 2]] - 5.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn brute_force_breaks_ties_by_the_smallest_training_index() {
    let training = array![[1.0_f64, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
    let queries = array![[0.0_f64, 0.0]];
    let searcher = SKBruteForceSearcher::build(
        SKTrainingData::Dense(training),
        SKNeighborsMetric::Euclidean,
    )
    .unwrap();

    let found = searcher
        .k_nearest(SKDataView::Dense(queries.view()), 3)
        .unwrap();

    assert_eq!(found.indices().row(0).to_vec(), vec![0, 1, 2]);
}

#[test]
fn brute_force_chunked_batches_match_a_single_batch() {
    let training = dense_training();
    let queries = Array2::from_shape_fn((13, 3), |(row, column)| {
        ((row * 3 + column * 5) % 9) as f64 / 4.0
    });
    let full = sk_brute_force_k_nearest_with_budget(
        SKDataView::Dense(training.view()),
        SKNeighborsMetric::Euclidean,
        SKDataView::Dense(queries.view()),
        4,
        1 << 24,
    )
    .unwrap();
    let chunked = sk_brute_force_k_nearest_with_budget(
        SKDataView::Dense(training.view()),
        SKNeighborsMetric::Euclidean,
        SKDataView::Dense(queries.view()),
        4,
        16,
    )
    .unwrap();

    assert_eq!(full.indices(), chunked.indices());
    assert_eq!(full.distances(), chunked.distances());
}
