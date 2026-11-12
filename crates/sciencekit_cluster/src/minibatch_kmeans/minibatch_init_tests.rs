//! Companion tests for bounded-sample initialization and best-of-`n_init`
//! selection (spec `mini-batch-k-means`, task 2.1).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKDataView};
use sprs::TriMat;

use super::assign::sk_assign_batch_rows;
use super::init::{sk_initialize_minibatch_centers, sk_sample_initialization_indices};
use super::{SKKMeansInitialization, SKMiniBatchKMeans};
use crate::kmeans::init::{sk_initialization_seed, sk_initialize_centers};

#[test]
fn minibatch_init_respects_initialization_batch_size_bound_on_dense_and_sparse() {
    let mut dataset = Array2::<f64>::zeros((40, 2));
    dataset
        .rows_mut()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, mut row)| {
            let base = if row_index % 2 == 0 { 0.0 } else { 50.0 };
            row[0] = base + (row_index as f64) * 0.1;
            row[1] = base - (row_index as f64) * 0.05;
        });

    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .initialization_batch_size(8)
        .number_of_initializations(3)
        .random_seed(17)
        .build()
        .unwrap();

    let sampled = sk_sample_initialization_indices(40, 8, 17);
    assert_eq!(sampled.len(), 8);
    assert!(sampled.windows(2).all(|pair| pair[0] < pair[1]));

    let (dense_centers, dense_inertia) = sk_initialize_minibatch_centers(
        SKDataView::Dense(dataset.view()),
        &estimator.configuration,
    )
    .unwrap();
    assert_eq!(dense_centers.dim(), (2, 2));
    assert!(dense_inertia.is_finite());

    // Mutating every unsampled row must not affect the initialized centers.
    let mut perturbed = dataset.clone();
    perturbed
        .rows_mut()
        .into_iter()
        .enumerate()
        .filter(|(row_index, _)| !sampled.contains(row_index))
        .for_each(|(_, mut row)| {
            row[0] = 1.0e9;
            row[1] = -1.0e9;
        });
    let (perturbed_centers, perturbed_inertia) = sk_initialize_minibatch_centers(
        SKDataView::Dense(perturbed.view()),
        &estimator.configuration,
    )
    .unwrap();
    assert_eq!(dense_centers, perturbed_centers);
    assert!((dense_inertia - perturbed_inertia).abs() < 1e-12);

    // CSR representation produces the same bounded initialization without densifying.
    let mut triplet = TriMat::new(dataset.dim());
    dataset
        .indexed_iter()
        .filter(|(_, value)| **value != 0.0)
        .for_each(|((row, column), &value)| triplet.add_triplet(row, column, value));
    let sparse = triplet.to_csr();
    let (sparse_centers, sparse_inertia) = sk_initialize_minibatch_centers(
        SKDataView::Sparse(sparse.view()),
        &estimator.configuration,
    )
    .unwrap();
    dense_centers
        .iter()
        .zip(sparse_centers.iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-12));
    assert!((dense_inertia - sparse_inertia).abs() < 1e-10);
}

#[test]
fn minibatch_init_selects_lowest_inertia_across_multiple_initializations() {
    let dataset = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.3, 0.0],
        [100.0, 100.0],
        [100.1, 100.0]
    ];

    let multi = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(6)
        .initialization_batch_size(6)
        .random_seed(5)
        .build()
        .unwrap();
    let (_, best_inertia) =
        sk_initialize_minibatch_centers(SKDataView::Dense(dataset.view()), &multi.configuration)
            .unwrap();

    let manual_minimum = (0..6)
        .map(|index| {
            let seed = sk_initialization_seed(5, index);
            let centers = sk_initialize_centers(
                SKDataView::Dense(dataset.view()),
                2,
                SKKMeansInitialization::Random,
                seed,
            )
            .unwrap();
            let (_, distances) =
                sk_assign_batch_rows(SKDataView::Dense(dataset.view()), &centers.view()).unwrap();
            distances.sum()
        })
        .fold(f64::INFINITY, f64::min);

    assert!((best_inertia - manual_minimum).abs() < 1e-12);

    let single = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .initialization(SKKMeansInitialization::Random)
        .number_of_initializations(1)
        .initialization_batch_size(6)
        .random_seed(5)
        .build()
        .unwrap();
    let (_, single_inertia) =
        sk_initialize_minibatch_centers(SKDataView::Dense(dataset.view()), &single.configuration)
            .unwrap();
    assert!(best_inertia <= single_inertia + 1e-12);
}
