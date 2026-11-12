//! Companion tests for dense/sparse representation parity and non-densification
//! (spec `mini-batch-k-means`, task 2.3).

use ndarray::{Array2, Axis, Slice, array};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKMiniBatchKMeans;
use super::assign::sk_assign_batch_rows;

fn to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    dense
        .indexed_iter()
        .filter(|(_, value)| **value != 0.0)
        .for_each(|((row, column), &value)| triplet.add_triplet(row, column, value));
    triplet.to_csr()
}

#[test]
fn minibatch_representation_csr_fit_equals_dense_fit_centers_counts_and_labels() {
    let dense = array![
        [1.0_f64, 0.0, 0.0, 2.0],
        [1.2, 0.0, 0.0, 1.9],
        [0.9, 0.0, 0.1, 2.1],
        [1.1, 0.0, 0.0, 2.0],
        [0.0, 15.0, 20.0, 0.0],
        [0.0, 15.2, 19.8, 0.0],
        [0.0, 14.9, 20.1, 0.0],
        [0.1, 15.1, 20.0, 0.0]
    ];
    let sparse = to_csr(&dense);

    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(4)
        .initialization_batch_size(6)
        .maximum_number_of_iterations(6)
        .random_seed(23)
        .build()
        .unwrap();

    let dense_model = estimator.fit(&dense).unwrap();
    let sparse_model = estimator.fit(sparse.view()).unwrap();

    assert_eq!(dense_model.counts(), sparse_model.counts());
    assert_eq!(dense_model.labels(), sparse_model.labels());
    assert_eq!(
        dense_model.batches_processed(),
        sparse_model.batches_processed()
    );
    assert_eq!(
        dense_model.epochs_consumed(),
        sparse_model.epochs_consumed()
    );
    assert!((dense_model.inertia() - sparse_model.inertia()).abs() < 1e-10);
    dense_model
        .cluster_centers()
        .iter()
        .zip(sparse_model.cluster_centers().iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-10));
}

#[test]
fn minibatch_representation_mixed_dense_and_csr_partial_fit_matches_uniform_paths() {
    let full = array![
        [2.0_f64, 0.0, 0.0],
        [2.2, 0.1, 0.0],
        [1.9, 0.0, 0.0],
        [0.0, 0.0, 30.0],
        [0.0, 0.2, 29.8],
        [0.1, 0.0, 30.1],
        [2.1, 0.0, 0.0],
        [1.8, 0.1, 0.0],
        [0.0, 0.0, 30.2],
        [0.0, 0.1, 29.9]
    ];
    let first_dense = full.slice_axis(Axis(0), Slice::from(0..6)).to_owned();
    let second_dense = full.slice_axis(Axis(0), Slice::from(6..10)).to_owned();
    let first_sparse = to_csr(&first_dense);
    let second_sparse = to_csr(&second_dense);

    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(6)
        .random_seed(9)
        .build()
        .unwrap();

    let mut all_dense = estimator.fit_batch(&first_dense).unwrap();
    all_dense.partial_fit(&second_dense).unwrap();

    let mut all_sparse = estimator.fit_batch(first_sparse.view()).unwrap();
    all_sparse.partial_fit(second_sparse.view()).unwrap();

    let mut mixed = estimator.fit_batch(&first_dense).unwrap();
    mixed.partial_fit(second_sparse.view()).unwrap();

    assert_eq!(all_dense.counts(), all_sparse.counts());
    assert_eq!(all_dense.counts(), mixed.counts());
    all_dense
        .cluster_centers()
        .iter()
        .zip(all_sparse.cluster_centers().iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-10));
    all_dense
        .cluster_centers()
        .iter()
        .zip(mixed.cluster_centers().iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-10));
}

#[test]
fn minibatch_representation_wide_csr_fits_without_densifying_batch() {
    let number_of_rows = 400_usize;
    let number_of_columns = 50_000_usize;
    let mut triplet = TriMat::new((number_of_rows, number_of_columns));
    (0..number_of_rows).for_each(|row| {
        if row % 2 == 0 {
            triplet.add_triplet(row, 3, 10.0_f64);
            triplet.add_triplet(row, 49_990, 5.0_f64);
        } else {
            triplet.add_triplet(row, 1_000, -12.0_f64);
            triplet.add_triplet(row, 25_000, 8.0_f64);
        }
    });
    let wide_sparse = triplet.to_csr();

    let model = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(64)
        .initialization_batch_size(64)
        .maximum_number_of_iterations(3)
        .random_seed(0)
        .build()
        .unwrap()
        .fit(wide_sparse.view())
        .unwrap();

    assert_eq!(model.number_of_clusters(), 2);
    assert_eq!(model.number_of_features(), number_of_columns);
    assert_eq!(model.labels().unwrap().len(), number_of_rows);
    let centers = model.cluster_centers();
    let even_cluster = model.labels().unwrap()[0].cluster_index().unwrap() as usize;
    let odd_cluster = model.labels().unwrap()[1].cluster_index().unwrap() as usize;
    assert_ne!(even_cluster, odd_cluster);
    assert!((centers[[even_cluster, 3]] - 10.0).abs() < 1e-6);
    assert!((centers[[odd_cluster, 1_000]] - (-12.0)).abs() < 1e-6);
}

#[test]
fn minibatch_representation_rejects_csc_sparse_view_up_front() {
    let csc = to_csr(&array![[1.0_f64, 0.0], [0.0, 2.0], [3.0, 0.0]]).to_csc();
    let centers = array![[0.0_f64, 0.0], [5.0, 5.0]];

    match sk_assign_batch_rows(SKDataView::Sparse(csc.view()), &centers.view()).unwrap_err() {
        SKError::UnsupportedRepresentation { representation, .. } => {
            assert_eq!(representation, "csc");
        }
        other => panic!("expected UnsupportedRepresentation(csc), got {other:?}"),
    }

    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .build()
        .unwrap();
    match estimator.fit(csc.view()).unwrap_err() {
        SKError::UnsupportedRepresentation { representation, .. } => {
            assert_eq!(representation, "csc");
        }
        other => panic!("expected UnsupportedRepresentation(csc), got {other:?}"),
    }

    let mut fitted = estimator
        .fit_batch(&array![[0.0_f64, 0.0], [10.0, 10.0], [10.1, 10.0]])
        .unwrap();
    match fitted.partial_fit(csc.view()).unwrap_err() {
        SKError::UnsupportedRepresentation { representation, .. } => {
            assert_eq!(representation, "csc");
        }
        other => panic!("expected UnsupportedRepresentation(csc), got {other:?}"),
    }
}
