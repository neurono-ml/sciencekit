//! Companion tests for bagging row and feature sampling and non-densifying
//! submatrix projections.

use ndarray::Array2;
use sciencekit_tree::SKMaximumFeatures;
use sprs::TriMat;

use crate::SKMaximumSamples;

use super::sampling::{
    sk_bagging_feature_indices, sk_bagging_row_indices, sk_gather_dense_submatrix,
    sk_gather_sparse_submatrix, sk_is_identity_features, sk_project_dense_rows,
    sk_project_sparse_rows,
};

#[test]
fn row_sampling_is_reproducible_per_estimator_and_honors_policies() {
    let first = sk_bagging_row_indices(16, &SKMaximumSamples::All, true, 42, 0);
    let repeated = sk_bagging_row_indices(16, &SKMaximumSamples::All, true, 42, 0);
    let next_estimator = sk_bagging_row_indices(16, &SKMaximumSamples::All, true, 42, 1);
    assert_eq!(first.len(), 16);
    assert_eq!(first, repeated);
    assert_ne!(first, next_estimator);

    let fractional = sk_bagging_row_indices(20, &SKMaximumSamples::Fraction(0.4), true, 11, 2);
    assert_eq!(fractional.len(), 8);
    assert!(fractional.iter().all(|&row| row < 20));

    let without_replacement = sk_bagging_row_indices(10, &SKMaximumSamples::Count(6), false, 11, 0);
    assert_eq!(without_replacement.len(), 6);
    let mut deduplicated = without_replacement.clone();
    deduplicated.sort_unstable();
    deduplicated.dedup();
    assert_eq!(deduplicated.len(), 6);
}

#[test]
fn feature_sampling_supports_identity_subspaces_and_bootstrap_with_replacement() {
    let all_no_bootstrap = sk_bagging_feature_indices(6, &SKMaximumFeatures::All, false, 7, 0);
    assert_eq!(all_no_bootstrap, vec![0, 1, 2, 3, 4, 5]);
    assert!(sk_is_identity_features(&all_no_bootstrap, 6));

    let subspace_a = sk_bagging_feature_indices(8, &SKMaximumFeatures::Fraction(0.5), false, 7, 0);
    let subspace_b = sk_bagging_feature_indices(8, &SKMaximumFeatures::Fraction(0.5), false, 7, 0);
    let subspace_other =
        sk_bagging_feature_indices(8, &SKMaximumFeatures::Fraction(0.5), false, 7, 1);
    assert_eq!(subspace_a.len(), 4);
    assert_eq!(subspace_a, subspace_b);
    assert_ne!(subspace_a, subspace_other);
    assert!(subspace_a.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(!sk_is_identity_features(&subspace_a, 8));

    let bootstrapped_a = sk_bagging_feature_indices(6, &SKMaximumFeatures::All, true, 19, 0);
    let bootstrapped_b = sk_bagging_feature_indices(6, &SKMaximumFeatures::All, true, 19, 0);
    assert_eq!(bootstrapped_a.len(), 6);
    assert_eq!(bootstrapped_a, bootstrapped_b);
    assert!(bootstrapped_a.iter().all(|&column| column < 6));
}

#[test]
fn dense_and_sparse_submatrix_gathers_agree_including_duplicate_feature_draws() {
    let dense = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.0, 2.0, 0.0, // row 0
            0.0, 3.0, 0.0, 4.0, // row 1
            5.0, 0.0, 0.0, 6.0, // row 2
            0.0, 7.0, 8.0, 0.0, // row 3
        ],
    )
    .unwrap();
    let mut triplets = TriMat::new((4, 4));
    dense
        .indexed_iter()
        .filter(|entry| *entry.1 != 0.0)
        .for_each(|((row, column), &value)| triplets.add_triplet(row, column, value));
    let sparse = triplets.to_csr::<usize>();

    // Duplicate and out-of-order feature draw `[3, 0, 3]` on rows `[2, 0, 3]`.
    let rows = [2usize, 0, 3];
    let features = [3usize, 0, 3];
    let dense_gathered = sk_gather_dense_submatrix(&dense.view(), &rows, &features);
    let sparse_gathered = sk_gather_sparse_submatrix(&sparse.view(), &rows, &features);

    assert_eq!(dense_gathered.dim(), (3, 3));
    assert_eq!((sparse_gathered.rows(), sparse_gathered.cols()), (3, 3));
    assert_eq!(dense_gathered, sparse_gathered.to_dense());

    let dense_projected = sk_project_dense_rows(&dense.view(), 1, 4, &[2, 1]);
    let sparse_projected = sk_project_sparse_rows(&sparse.view(), 1, 4, &[2, 1]);
    assert_eq!(dense_projected, sparse_projected.to_dense());
}
