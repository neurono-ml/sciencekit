//! Parallel-equals-sequential acceptance for one-vs-rest head fitting
//! (PRD §8.7, spec `linear-support-vector-classifier`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use crate::SKLinearSVC;

fn dense_multiclass_fixture() -> (Array2<f64>, Array1<i64>) {
    let rows = 300usize;
    let features = Array2::from_shape_fn((rows, 40), |(row, column)| {
        let class = row % 3;
        let strong = if column == class { 2.0 } else { 0.0 };
        let noise = if column >= 10 && (row * 13 + column * 7) % 11 == 0 {
            0.3
        } else {
            0.0
        };
        strong + noise
    });
    let labels = Array1::from_shape_fn(rows, |row| (row % 3) as i64);
    (features, labels)
}

#[test]
fn acceptance_parallel_head_fitting_matches_a_single_threaded_run() {
    let (features, labels) = dense_multiclass_fixture();

    let parallel_model = SKLinearSVC::<f64>::build()
        .tolerance(1e-2)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let sequential_model = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            SKLinearSVC::<f64>::build()
                .tolerance(1e-2)
                .build()
                .unwrap()
                .fit(features.view(), labels.view())
                .unwrap()
        });

    assert_eq!(
        parallel_model.predict_labels(features.view()).unwrap(),
        sequential_model.predict_labels(features.view()).unwrap()
    );
    parallel_model
        .heads()
        .iter()
        .zip(sequential_model.heads().iter())
        .for_each(|(left, right)| {
            left.coefficients()
                .iter()
                .zip(right.coefficients().iter())
                .for_each(|(a, b)| assert!((a - b).abs() < 1e-12));
        });
}

#[test]
fn acceptance_dense_and_sparse_batches_predict_identically() {
    use sprs::TriMat;

    let (features, labels) = dense_multiclass_fixture();
    let mut triplets = TriMat::new(features.dim());
    features
        .indexed_iter()
        .filter(|(_, value)| **value != 0.0)
        .for_each(|((row, column), &value)| triplets.add_triplet(row, column, value));
    let sparse = triplets.to_csr();

    let model = SKLinearSVC::<f64>::build()
        .tolerance(1e-2)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        model.predict_labels(sparse.view()).unwrap()
    );
}
