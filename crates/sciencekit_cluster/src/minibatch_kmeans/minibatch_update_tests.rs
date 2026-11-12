//! Companion tests for per-center batch updates and count bookkeeping (spec
//! `mini-batch-k-means`, task 1.2).

use ndarray::{Array1, array};
use sciencekit_common::SKDataView;
use sprs::TriMat;

use super::assign::sk_assign_batch_rows;
use super::update::sk_update_centers_and_counts;

#[test]
fn minibatch_update_one_center_closed_form_moves_to_batch_mean_and_sets_count() {
    let mut centers = array![[0.5_f64, -0.5], [100.0, 100.0]];
    let mut counts = Array1::from_vec(vec![0_usize, 0]);
    let batch = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let (assignments, _) =
        sk_assign_batch_rows(SKDataView::Dense(batch.view()), &centers.view()).unwrap();
    assert!(assignments.iter().all(|&cluster| cluster == 0));

    let shift = sk_update_centers_and_counts(
        &mut centers,
        &mut counts,
        SKDataView::Dense(batch.view()),
        &assignments,
    )
    .unwrap();

    assert!((centers[[0, 0]] - 4.0).abs() < 1e-12);
    assert!((centers[[0, 1]] - 5.0).abs() < 1e-12);
    assert!((centers[[1, 0]] - 100.0).abs() < 1e-12);
    assert!((centers[[1, 1]] - 100.0).abs() < 1e-12);
    assert_eq!(counts.to_vec(), vec![4, 0]);
    let expected_shift = ((4.0_f64 - 0.5).powi(2) + (5.0_f64 - (-0.5)).powi(2)).sqrt();
    assert!((shift - expected_shift).abs() < 1e-12);
}

#[test]
fn minibatch_update_learning_rate_decays_as_one_over_n_plus_one() {
    for prior_count in [0_usize, 1, 4, 9, 24] {
        let initial_center = [2.0_f64, -4.0];
        let new_row = [12.0_f64, 6.0];
        let mut centers = array![initial_center, [1000.0, 1000.0]];
        let mut counts = Array1::from_vec(vec![prior_count, 5]);
        let batch = array![new_row];

        let (assignments, _) =
            sk_assign_batch_rows(SKDataView::Dense(batch.view()), &centers.view()).unwrap();
        assert_eq!(assignments.to_vec(), vec![0]);

        sk_update_centers_and_counts(
            &mut centers,
            &mut counts,
            SKDataView::Dense(batch.view()),
            &assignments,
        )
        .unwrap();

        let weight = 1.0 / (prior_count as f64 + 1.0);
        let expected_x = initial_center[0] + weight * (new_row[0] - initial_center[0]);
        let expected_y = initial_center[1] + weight * (new_row[1] - initial_center[1]);
        assert!((centers[[0, 0]] - expected_x).abs() < 1e-12);
        assert!((centers[[0, 1]] - expected_y).abs() < 1e-12);
        assert_eq!(counts[0], prior_count + 1);
        assert_eq!(counts[1], 5);
    }
}

#[test]
fn minibatch_update_sequential_single_rows_equal_batch_update_and_sparse_batch() {
    let batch = array![[1.0_f64, 0.0, 3.0], [0.0, 4.0, 0.0], [11.0, 0.0, 9.0]];
    let initial_centers = array![[0.2_f64, 0.1, 0.3], [10.0, 0.0, 10.0]];

    let mut dense_centers = initial_centers.clone();
    let mut dense_counts = Array1::from_vec(vec![2_usize, 3]);
    let (assignments, _) =
        sk_assign_batch_rows(SKDataView::Dense(batch.view()), &dense_centers.view()).unwrap();
    sk_update_centers_and_counts(
        &mut dense_centers,
        &mut dense_counts,
        SKDataView::Dense(batch.view()),
        &assignments,
    )
    .unwrap();

    let mut row_by_row_centers = initial_centers.clone();
    let mut row_by_row_counts = Array1::from_vec(vec![2_usize, 3]);
    for (row, &cluster) in batch.rows().into_iter().zip(assignments.iter()) {
        let single_row = row.insert_axis(ndarray::Axis(0));
        let single_assignment = Array1::from_vec(vec![cluster]);
        sk_update_centers_and_counts(
            &mut row_by_row_centers,
            &mut row_by_row_counts,
            SKDataView::Dense(single_row),
            &single_assignment,
        )
        .unwrap();
    }

    let mut triplet = TriMat::new(batch.dim());
    for ((row, column), &value) in batch.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    let sparse_batch = triplet.to_csr();
    let mut sparse_centers = initial_centers;
    let mut sparse_counts = Array1::from_vec(vec![2_usize, 3]);
    let (sparse_assignments, _) = sk_assign_batch_rows(
        SKDataView::Sparse(sparse_batch.view()),
        &sparse_centers.view(),
    )
    .unwrap();
    assert_eq!(sparse_assignments, assignments);
    sk_update_centers_and_counts(
        &mut sparse_centers,
        &mut sparse_counts,
        SKDataView::Sparse(sparse_batch.view()),
        &sparse_assignments,
    )
    .unwrap();

    assert_eq!(dense_counts, row_by_row_counts);
    assert_eq!(dense_counts, sparse_counts);
    dense_centers
        .iter()
        .zip(row_by_row_centers.iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-12));
    dense_centers
        .iter()
        .zip(sparse_centers.iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-12));
}
