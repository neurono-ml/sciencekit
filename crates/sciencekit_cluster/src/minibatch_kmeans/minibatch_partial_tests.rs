//! Companion tests for incremental `partial_fit` across batches (spec
//! `mini-batch-k-means`, task 2.2).

use ndarray::{Array2, Axis, array, concatenate};
use sciencekit_common::{SKBuilder, SKDataView, SKUnsupervisedFit};

use super::SKMiniBatchKMeans;
use super::assign::sk_assign_batch_rows;

fn planted_batches() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let batch_one = array![
        [0.0_f64, 0.0],
        [0.2, -0.1],
        [0.1, 0.2],
        [20.0, 20.0],
        [20.2, 19.9],
        [19.9, 20.1]
    ];
    let batch_two = array![
        [-0.1_f64, 0.1],
        [0.3, 0.0],
        [0.0, -0.2],
        [20.1, 20.2],
        [19.8, 20.0],
        [20.0, 19.8]
    ];
    let batch_three = array![
        [0.1_f64, 0.1],
        [-0.2, 0.0],
        [0.2, 0.2],
        [20.3, 20.1],
        [19.9, 19.9],
        [20.1, 20.0]
    ];
    (batch_one, batch_two, batch_three)
}

#[test]
fn minibatch_partial_persists_centers_counts_and_batches_across_chained_calls() {
    let (batch_one, batch_two, batch_three) = planted_batches();
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(6)
        .reassignment_ratio(0.0)
        .random_seed(42)
        .build()
        .unwrap();

    let mut model = estimator.fit_batch(&batch_one).unwrap();
    let first_counts = model.counts().to_owned();
    let first_centers = model.cluster_centers().to_owned();
    assert_eq!(first_counts.sum(), 6);
    assert_eq!(model.batches_processed(), 1);
    assert!(model.labels().is_none());

    model.partial_fit(&batch_two).unwrap();
    let second_counts = model.counts().to_owned();
    assert_eq!(second_counts.sum(), 12);
    assert!(
        second_counts
            .iter()
            .zip(first_counts.iter())
            .all(|(&after, &before)| after >= before)
    );
    assert_ne!(model.cluster_centers(), first_centers.view());
    assert_eq!(model.batches_processed(), 2);

    model.partial_fit(&batch_three).unwrap();
    assert_eq!(model.counts().sum(), 18);
    assert_eq!(model.batches_processed(), 3);
}

#[test]
fn minibatch_partial_batched_updates_agree_with_full_fit_on_concatenated_data() {
    let (batch_one, batch_two, batch_three) = planted_batches();
    let concatenated = concatenate(
        Axis(0),
        &[batch_one.view(), batch_two.view(), batch_three.view()],
    )
    .unwrap();

    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(6)
        .initialization_batch_size(6)
        .maximum_number_of_iterations(5)
        .random_seed(11)
        .build()
        .unwrap();

    let full_model = estimator.fit(&concatenated).unwrap();
    assert!(full_model.labels().is_some());
    assert_eq!(full_model.labels().unwrap().len(), 18);

    let mut incremental = estimator.fit_batch(&batch_one).unwrap();
    incremental
        .partial_fit(&batch_two)
        .unwrap()
        .partial_fit(&batch_three)
        .unwrap();
    for _ in 1..5 {
        incremental
            .partial_fit(&batch_one)
            .unwrap()
            .partial_fit(&batch_two)
            .unwrap()
            .partial_fit(&batch_three)
            .unwrap();
    }

    // Sort centers by first coordinate to compare regardless of cluster permutation.
    let mut full_rows: Vec<[f64; 2]> = full_model
        .cluster_centers()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1]])
        .collect();
    let mut incremental_rows: Vec<[f64; 2]> = incremental
        .cluster_centers()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1]])
        .collect();
    full_rows.sort_by(|left, right| left[0].partial_cmp(&right[0]).unwrap());
    incremental_rows.sort_by(|left, right| left[0].partial_cmp(&right[0]).unwrap());

    full_rows
        .iter()
        .zip(incremental_rows.iter())
        .for_each(|(left, right)| {
            assert!((left[0] - right[0]).abs() < 0.25);
            assert!((left[1] - right[1]).abs() < 0.25);
        });

    let (full_assignments, _) = sk_assign_batch_rows(
        SKDataView::Dense(concatenated.view()),
        &full_model.cluster_centers(),
    )
    .unwrap();
    let (incremental_assignments, _) = sk_assign_batch_rows(
        SKDataView::Dense(concatenated.view()),
        &incremental.cluster_centers(),
    )
    .unwrap();

    // Every sample in [0, 3), [6, 9), [12, 15) shares one cluster and [3, 6), [9, 12), [15, 18) shares the other.
    let same_or_flipped = full_assignments == incremental_assignments
        || full_assignments
            .iter()
            .zip(incremental_assignments.iter())
            .all(|(&left, &right)| left + right == 1);
    assert!(same_or_flipped);
}
