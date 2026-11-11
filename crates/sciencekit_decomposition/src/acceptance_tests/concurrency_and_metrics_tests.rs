//! Acceptance tests for determinism, multi-threaded concurrency and metric
//! production through `sciencekit_metrics`.

use std::sync::Arc;

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};
use sciencekit_metrics::regression::{sk_mean_squared_error, sk_r2_score};

use super::fixtures::planted_rank_matrix;
use crate::SKPCA;

#[test]
fn acceptance_pca_determinism_and_concurrent_sharing_on_large_data() {
    let data = planted_rank_matrix(1_000, 30, 4, 1e-3);
    let estimator = SKPCA::<f64>::build()
        .number_of_components(4)
        .whiten(true)
        .random_seed(77)
        .build()
        .unwrap();

    let model_a = estimator.fit(&data).unwrap();
    let model_b = estimator.fit(&data).unwrap();
    assert_eq!(model_a.components(), model_b.components());
    assert_eq!(model_a.explained_variance(), model_b.explained_variance());

    let shared = Arc::new(model_a);
    let reference = shared.transform(&data).unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let model = Arc::clone(&shared);
                let data_ref = &data;
                scope.spawn(move || model.transform(data_ref).unwrap())
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), reference);
        }
    });
}

#[test]
fn acceptance_pca_produces_reconstruction_metrics_via_sciencekit_metrics() {
    let data = planted_rank_matrix(600, 16, 3, 1e-3);
    let flattened_truth = Array1::from_iter(data.iter().copied());

    let mut previous_mse = f64::INFINITY;
    for components in [1usize, 2, 3] {
        let model = SKPCA::<f64>::build()
            .number_of_components(components)
            .random_seed(13)
            .build()
            .unwrap()
            .fit(&data)
            .unwrap();
        let projected = model.transform(&data).unwrap();
        let reconstructed = model.inverse_transform(&projected).unwrap();
        let flattened_reconstructed = Array1::from_iter(reconstructed.iter().copied());

        let mse =
            sk_mean_squared_error(flattened_truth.view(), flattened_reconstructed.view()).unwrap();
        let r2 = sk_r2_score(flattened_truth.view(), flattened_reconstructed.view()).unwrap();
        assert!(mse < previous_mse);
        previous_mse = mse;

        if components == 3 {
            assert!(mse < 1e-5);
            assert!(r2 > 0.9999);
        }
    }
}
