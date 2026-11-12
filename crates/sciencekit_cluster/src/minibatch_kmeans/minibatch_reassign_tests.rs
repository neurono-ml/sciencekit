//! Companion tests for deterministic low-count center reassignment (spec
//! `mini-batch-k-means`, task 1.3).

use ndarray::{Array1, array};
use sciencekit_common::SKDataView;
use sprs::TriMat;

use super::reassign::sk_reassign_low_count_centers;

#[test]
fn minibatch_reassign_moves_distant_outlier_to_low_count_center_and_resets_count() {
    let mut centers = array![[0.0_f64, 0.0], [999.0, 999.0]];
    let mut counts = Array1::from_vec(vec![100_usize, 0]);
    let batch = array![[0.1_f64, 0.0], [-0.1, 0.2], [25.0, -15.0], [0.0, -0.1]];
    let assignments = Array1::from_vec(vec![0_usize, 0, 0, 0]);

    let reassigned = sk_reassign_low_count_centers(
        &mut centers,
        &mut counts,
        SKDataView::Dense(batch.view()),
        &assignments,
        0.01,
    )
    .unwrap();

    assert_eq!(reassigned, 1);
    assert!((centers[[1, 0]] - 25.0).abs() < 1e-12);
    assert!((centers[[1, 1]] - (-15.0)).abs() < 1e-12);
    assert_eq!(counts[1], 1);
    assert_eq!(counts[0], 100);
}

#[test]
fn minibatch_reassign_skips_centers_at_or_above_threshold_and_zero_ratio() {
    let mut centers = array![[0.0_f64, 0.0], [999.0, 999.0]];
    let mut counts = Array1::from_vec(vec![99_usize, 1]);
    let batch = array![[0.0_f64, 0.0], [50.0, 50.0]];
    let assignments = Array1::from_vec(vec![0_usize, 0]);

    // mean_count = 50.0, threshold = 0.02 * 50.0 = 1.0; count == 1 is not strictly below 1.0
    let count_at_threshold = sk_reassign_low_count_centers(
        &mut centers,
        &mut counts,
        SKDataView::Dense(batch.view()),
        &assignments,
        0.02,
    )
    .unwrap();
    assert_eq!(count_at_threshold, 0);
    assert!((centers[[1, 0]] - 999.0).abs() < 1e-12);

    // Zero ratio never reassigns even a zero-count center.
    let mut zero_counts = Array1::from_vec(vec![100_usize, 0]);
    let zero_ratio = sk_reassign_low_count_centers(
        &mut centers,
        &mut zero_counts,
        SKDataView::Dense(batch.view()),
        &assignments,
        0.0,
    )
    .unwrap();
    assert_eq!(zero_ratio, 0);
    assert_eq!(zero_counts[1], 0);
}

#[test]
fn minibatch_reassign_breaks_ties_by_smallest_row_index_on_dense_and_sparse() {
    let batch = array![[0.0_f64, 1.0], [10.0, 0.0], [0.0, -1.0], [-10.0, 0.0]];
    let assignments = Array1::from_vec(vec![0_usize, 0, 0, 0]);

    let mut dense_centers = array![[0.0_f64, 0.0], [500.0, 500.0]];
    let mut dense_counts = Array1::from_vec(vec![50_usize, 0]);
    sk_reassign_low_count_centers(
        &mut dense_centers,
        &mut dense_counts,
        SKDataView::Dense(batch.view()),
        &assignments,
        0.05,
    )
    .unwrap();

    let mut triplet = TriMat::new(batch.dim());
    for ((row, column), &value) in batch.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    let sparse_batch = triplet.to_csr();
    let mut sparse_centers = array![[0.0_f64, 0.0], [500.0, 500.0]];
    let mut sparse_counts = Array1::from_vec(vec![50_usize, 0]);
    sk_reassign_low_count_centers(
        &mut sparse_centers,
        &mut sparse_counts,
        SKDataView::Sparse(sparse_batch.view()),
        &assignments,
        0.05,
    )
    .unwrap();

    // Row 1 ([10, 0]) and row 3 ([-10, 0]) tie at squared distance 100; row 1 wins.
    assert_eq!(dense_centers.row(1), array![10.0_f64, 0.0].view());
    assert_eq!(sparse_centers.row(1), array![10.0_f64, 0.0].view());
    assert_eq!(dense_counts[1], 1);
    assert_eq!(sparse_counts[1], 1);
}
