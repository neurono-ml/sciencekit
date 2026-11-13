//! Companion tests for the Lloyd loop and empty-cluster repair.

use ndarray::{Array1, Array2, array};
use sciencekit_common::SKDataView;
use sprs::{CsMat, TriMat};

use super::iteration::sk_update_centers;
use super::lloyd::sk_lloyd_fit;

fn separated_blobs() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1]
    ]
}

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

#[test]
fn kmeans_lloyd_two_well_separated_clusters_converge() {
    let data = separated_blobs();
    let initial_centers = array![[0.0_f64, 0.0], [10.0, 10.0]];
    let result = sk_lloyd_fit(SKDataView::Dense(data.view()), initial_centers, 100, 1e-8).unwrap();

    assert!(result.converged);
    assert!(result.iterations >= 1);
    let first_cluster = result.labels[0];
    for row in 0..3 {
        assert_eq!(result.labels[row], first_cluster);
    }
    for row in 3..6 {
        assert_ne!(result.labels[row], first_cluster);
    }
    assert!(result.inertia < 0.05);
}

#[test]
fn kmeans_lloyd_exhausted_budget_returns_an_unconverged_model() {
    let data = array![[0.0_f64], [1.0], [10.0], [11.0]];
    let initial_centers = array![[0.0_f64], [11.0]];
    let result = sk_lloyd_fit(SKDataView::Dense(data.view()), initial_centers, 1, 1e-4).unwrap();

    assert!(!result.converged);
    assert_eq!(result.iterations, 1);
    assert!((result.inertia - 1.0).abs() < 1e-12);
}

#[test]
fn kmeans_lloyd_empty_cluster_repair_moves_the_farthest_point() {
    let data = array![[0.0_f64], [0.1], [5.0], [10.0]];
    let labels = Array1::from(vec![0_usize, 0, 1, 1]);
    let assigned_distances = Array1::from(vec![0.0_f64, 0.01, 0.0, 25.0]);

    let centers = sk_update_centers(
        SKDataView::Dense(data.view()),
        &labels,
        &assigned_distances,
        3,
    );
    let repeated = sk_update_centers(
        SKDataView::Dense(data.view()),
        &labels,
        &assigned_distances,
        3,
    );

    assert_eq!(centers, repeated);
    assert!((centers[[0, 0]] - 0.05).abs() < 1e-12);
    assert!((centers[[1, 0]] - 5.0).abs() < 1e-12);
    assert!((centers[[2, 0]] - 10.0).abs() < 1e-12);
}

#[test]
fn kmeans_lloyd_empty_cluster_repair_breaks_ties_by_row_index() {
    let data = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let labels = Array1::from(vec![0_usize, 0, 1, 1]);
    let assigned_distances = Array1::from(vec![0.0_f64, 0.0, 4.0, 4.0]);

    let centers = sk_update_centers(
        SKDataView::Dense(data.view()),
        &labels,
        &assigned_distances,
        3,
    );
    assert!((centers[[2, 0]] - 2.0).abs() < 1e-12);
}

#[test]
fn kmeans_lloyd_sparse_matches_dense() {
    let dense = separated_blobs();
    let sparse = csr_from_dense(&dense);
    let initial_centers = array![[0.0_f64, 0.0], [10.0, 10.0]];

    let dense_result = sk_lloyd_fit(
        SKDataView::Dense(dense.view()),
        initial_centers.clone(),
        100,
        1e-8,
    )
    .unwrap();
    let sparse_result = sk_lloyd_fit(
        SKDataView::Sparse(sparse.view()),
        initial_centers,
        100,
        1e-8,
    )
    .unwrap();

    assert_eq!(dense_result.labels, sparse_result.labels);
    for (dense_value, sparse_value) in dense_result
        .centers
        .iter()
        .zip(sparse_result.centers.iter())
    {
        assert!((dense_value - sparse_value).abs() < 1e-12);
    }
    assert!((dense_result.inertia - sparse_result.inertia).abs() < 1e-12);
}
