//! Companion tests for dense/sparse isolation tree parity without densification.

use ndarray::{Array2, array};
use sciencekit_common::SKRandomGenerator;
use sprs::{CsMat, TriMat};

use super::score::{sk_anomaly_score_from_expected_depth, sk_expected_path_length};
use super::sparse::{SparseSubsample, sk_gather_dense_rows, sk_sparse_row_value};
use super::tree::{
    sk_build_dense_isolation_tree, sk_build_sparse_isolation_tree, sk_isolation_depth_limit,
};

fn dense_to_csr(matrix: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((matrix.nrows(), matrix.ncols()));
    matrix
        .rows()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, row_values)| {
            row_values
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0.0)
                .for_each(|(column_index, &value)| {
                    triplets.add_triplet(row_index, column_index, value);
                });
        });
    triplets.to_csr()
}

#[test]
fn isolation_representation_sparse_and_dense_trees_and_scores_match_exactly() {
    // Includes negative entries, positive entries, and implicit zeros.
    let dense = array![
        [0.0_f64, -1.0, 0.0, 2.0],
        [0.2, 0.0, 0.0, 1.8],
        [-0.1, -0.8, 0.0, 2.1],
        [0.0, -1.1, 0.3, 1.9],
        [0.1, -0.9, 0.0, 2.2],
        [0.0, 0.0, 0.0, 2.0],
        [-15.0, 22.0, -9.0, 0.0],
        [0.0, -1.0, 0.1, 2.0]
    ];
    let csr = dense_to_csr(&dense);
    let sampled_rows: Vec<usize> = (0..dense.nrows()).collect();
    let candidate_features = [0usize, 1, 2, 3];
    let depth_limit = sk_isolation_depth_limit(sampled_rows.len());

    let dense_trees: Vec<_> = (0..16_u64)
        .map(|seed| {
            let mut generator = SKRandomGenerator::new(100 + seed);
            let gathered = sk_gather_dense_rows(dense.view(), &sampled_rows);
            sk_build_dense_isolation_tree(
                gathered.view(),
                &candidate_features,
                depth_limit,
                &mut generator,
            )
        })
        .collect();

    let sparse_trees: Vec<_> = (0..16_u64)
        .map(|seed| {
            let mut generator = SKRandomGenerator::new(100 + seed);
            sk_build_sparse_isolation_tree(
                csr.view(),
                &sampled_rows,
                &candidate_features,
                depth_limit,
                &mut generator,
            )
        })
        .collect();

    assert_eq!(dense_trees, sparse_trees);

    // Evaluate all four combinations: dense/sparse trees x dense/sparse query rows.
    (0..dense.nrows()).for_each(|row_index| {
        let sparse_row = csr.outer_view(row_index).unwrap();
        let depth_dense_dense =
            sk_expected_path_length(&dense_trees, |feature| dense[[row_index, feature]]);
        let depth_dense_sparse = sk_expected_path_length(&dense_trees, |feature| {
            sk_sparse_row_value(sparse_row, feature)
        });
        let depth_sparse_dense =
            sk_expected_path_length(&sparse_trees, |feature| dense[[row_index, feature]]);
        let depth_sparse_sparse = sk_expected_path_length(&sparse_trees, |feature| {
            sk_sparse_row_value(sparse_row, feature)
        });

        let score_dense_dense =
            sk_anomaly_score_from_expected_depth(depth_dense_dense, sampled_rows.len());
        let score_dense_sparse =
            sk_anomaly_score_from_expected_depth(depth_dense_sparse, sampled_rows.len());
        let score_sparse_dense =
            sk_anomaly_score_from_expected_depth(depth_sparse_dense, sampled_rows.len());
        let score_sparse_sparse =
            sk_anomaly_score_from_expected_depth(depth_sparse_sparse, sampled_rows.len());

        assert!((score_dense_dense - score_dense_sparse).abs() < 1e-15);
        assert!((score_dense_dense - score_sparse_dense).abs() < 1e-15);
        assert!((score_dense_dense - score_sparse_sparse).abs() < 1e-15);
    });
}

#[test]
fn isolation_representation_wide_sparse_subsample_never_densifies() {
    let row_count = 64usize;
    let column_count = 10_000usize;
    let mut triplets = TriMat::new((row_count, column_count));
    (0..row_count).for_each(|row_index| {
        triplets.add_triplet(row_index, row_index * 7, 1.5_f64);
    });
    // Plant an outlier at row 63 with large coordinates on two sparse columns.
    triplets.add_triplet(63, 9_999, 50.0_f64);
    let csr = triplets.to_csr();

    let sampled_rows: Vec<usize> = (0..row_count).collect();
    let subsample = SparseSubsample::from_csr_rows(csr.view(), &sampled_rows);
    assert_eq!(subsample.stored_entry_count(), row_count + 1);

    let candidate_features: Vec<usize> = (0..column_count).collect();
    let mut generator = SKRandomGenerator::new(2026);
    let tree = sk_build_sparse_isolation_tree(
        csr.view(),
        &sampled_rows,
        &candidate_features,
        sk_isolation_depth_limit(row_count),
        &mut generator,
    );
    assert!(tree.node_count() >= 3);
}
