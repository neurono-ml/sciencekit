//! Companion tests for the dense-only index contract.

use ndarray::array;
use sciencekit_common::{SKDataView, SKError};
use sprs::{CsMat, TriMat};

use crate::ball_tree::SKBallTreeBuilder;
use crate::kd_tree::SKKDTreeBuilder;
use crate::searcher::SKNearestNeighborsSearcher;

fn sparse() -> CsMat<f64> {
    let mut triplet = TriMat::new((3, 2));
    triplet.add_triplet(0, 0, 1.0);
    triplet.add_triplet(1, 1, 2.0);
    triplet.add_triplet(2, 0, 3.0);
    triplet.to_csr()
}

fn dense() -> ndarray::Array2<f64> {
    array![[1.0_f64, 0.0], [0.0, 2.0], [3.0, 0.0]]
}

fn assert_sparse_rejection(error: SKError) {
    match error {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csr");
            assert!(
                suggestion.contains("brute-force"),
                "the suggestion must name the brute-force path: {suggestion}"
            );
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}

#[test]
fn representation_rejects_sparse_training_for_both_trees() {
    let sparse = sparse();
    assert_sparse_rejection(
        SKKDTreeBuilder::<f64>::new()
            .build(SKDataView::Sparse(sparse.view()))
            .unwrap_err(),
    );
    assert_sparse_rejection(
        SKBallTreeBuilder::<f64>::new()
            .build(SKDataView::Sparse(sparse.view()))
            .unwrap_err(),
    );
}

#[test]
fn representation_rejects_sparse_queries_without_densifying() {
    let sparse = sparse();
    let dense = dense();
    let nnz_before = sparse.nnz();

    let kd = SKKDTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(dense.view()))
        .unwrap();
    assert_sparse_rejection(
        kd.k_nearest(SKDataView::Sparse(sparse.view()), 1)
            .unwrap_err(),
    );
    assert_sparse_rejection(
        kd.radius_neighbors(SKDataView::Sparse(sparse.view()), 1.0)
            .unwrap_err(),
    );

    let ball = SKBallTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(dense.view()))
        .unwrap();
    assert_sparse_rejection(
        ball.k_nearest(SKDataView::Sparse(sparse.view()), 1)
            .unwrap_err(),
    );

    assert_eq!(sparse.nnz(), nnz_before);
}
