//! End-to-end acceptance tests for `SKTruncatedSVD` covering tiny/large
//! dense and sparse CSR matrices, component boundaries, determinism,
//! multi-threaded concurrency and metric production (`spec` `truncated-svd`).

use std::sync::Arc;

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};
use sciencekit_metrics::regression::{sk_mean_squared_error, sk_r2_score};
use sprs::{CsMat, TriMat};

use super::fixtures::{planted_rank_matrix, planted_wide_csr};
use crate::{SK_TRUNCATED_SVD_PARITY_TOLERANCE, SKTruncatedSVD};

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
fn acceptance_truncated_svd_tiny_matrices_f32_and_component_boundaries() {
    // 1 x 3 degenerate single-row boundary (uncentered reconstruction still exact).
    let single_row = array![[3.0_f64, -4.0, 0.0]];
    let single_sparse = csr_from_dense(&single_row);
    for single_model in [
        SKTruncatedSVD::<f64>::build()
            .number_of_components(1)
            .build()
            .unwrap()
            .fit(&single_row)
            .unwrap(),
        SKTruncatedSVD::<f64>::build()
            .number_of_components(1)
            .build()
            .unwrap()
            .fit(single_sparse.view())
            .unwrap(),
    ] {
        assert_eq!(single_model.number_of_components(), 1);
        assert_eq!(single_model.explained_variance()[0], 0.0);
        assert_eq!(single_model.explained_variance_ratio()[0], 0.0);
        let rebuilt = single_model
            .inverse_transform(&single_model.transform(&single_row).unwrap())
            .unwrap();
        for (actual, expected) in rebuilt.iter().zip(single_row.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    // Single-precision f32 dense and CSR end-to-end.
    let data_f32 = array![
        [1.0_f32, 0.0, 2.0],
        [0.0, 3.0, 1.5],
        [2.0, 1.0, 4.0],
        [3.0, 2.0, 5.5]
    ];
    let mut triplet_f32 = TriMat::new(data_f32.dim());
    for ((row, column), &value) in data_f32.indexed_iter() {
        if value != 0.0 {
            triplet_f32.add_triplet(row, column, value);
        }
    }
    let csr_f32 = triplet_f32.to_csr();
    let model_f32 = SKTruncatedSVD::<f32>::build()
        .number_of_components(2)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(csr_f32.view())
        .unwrap();
    let projected_f32 = model_f32.transform(&data_f32).unwrap();
    assert_eq!(projected_f32.dim(), (4, 2));
    assert!(projected_f32.iter().all(|value| value.is_finite()));

    // Tall (12 x 4) and wide (4 x 9) component boundaries: k = 1 and k = min(n, d) = 4.
    let tall = Array2::from_shape_fn((12, 4), |(row, column)| {
        (row as f64 + 1.0) * (column as f64 + 1.5) + ((row + column) % 3) as f64 * 0.5
    });
    assert_eq!(
        SKTruncatedSVD::<f64>::build()
            .number_of_components(1)
            .build()
            .unwrap()
            .fit_transform(&tall)
            .unwrap()
            .dim(),
        (12, 1)
    );
    let tall_full = SKTruncatedSVD::<f64>::build()
        .number_of_components(4)
        .build()
        .unwrap()
        .fit(&tall)
        .unwrap();
    let tall_rebuilt = tall_full
        .inverse_transform(&tall_full.transform(&tall).unwrap())
        .unwrap();
    for (left, right) in tall_rebuilt.iter().zip(tall.iter()) {
        assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }

    let wide = Array2::from_shape_fn((4, 9), |(row, column)| {
        ((row + 1) as f64 * 0.7 * (column as f64 + 0.3)).sin() * 5.0 + column as f64 + 2.0
    });
    let wide_full = SKTruncatedSVD::<f64>::build()
        .number_of_components(4)
        .build()
        .unwrap()
        .fit(&wide)
        .unwrap();
    let wide_rebuilt = wide_full
        .inverse_transform(&wide_full.transform(&wide).unwrap())
        .unwrap();
    for (left, right) in wide_rebuilt.iter().zip(wide.iter()) {
        assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }
}

#[test]
fn acceptance_truncated_svd_large_dense_and_wide_sparse_concurrency_and_metrics() {
    // Large wide sparse CSR (2_000 x 10_000) without densification.
    let wide_sparse = planted_wide_csr(2_000, 10_000);
    let sparse_estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(4)
        .random_seed(33)
        .build()
        .unwrap();
    let sparse_model_a = sparse_estimator.fit(wide_sparse.view()).unwrap();
    let sparse_model_b = sparse_estimator.fit(wide_sparse.view()).unwrap();
    assert_eq!(sparse_model_a.components(), sparse_model_b.components());
    assert!(sparse_model_a.explained_variance_ratio().sum() <= 1.0);

    let shared_sparse_model = Arc::new(sparse_model_a);
    let reference_projection = shared_sparse_model.transform(wide_sparse.view()).unwrap();
    assert_eq!(reference_projection.dim(), (2_000, 4));
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let model = Arc::clone(&shared_sparse_model);
                let view = wide_sparse.view();
                scope.spawn(move || model.transform(view).unwrap())
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), reference_projection);
        }
    });

    // Uncentered planted rank-4 matrix (column offset + 3 waves => rank <= 4):
    // evaluate MSE and R^2 via sciencekit_metrics across k = 1..=4.
    let data = planted_rank_matrix(800, 20, 3, 1e-4);
    let flattened_truth = Array1::from_iter(data.iter().copied());
    let mut previous_mean_squared_error = f64::INFINITY;
    for components in 1..=4 {
        let model = SKTruncatedSVD::<f64>::build()
            .number_of_components(components)
            .random_seed(17)
            .build()
            .unwrap()
            .fit(&data)
            .unwrap();
        let reconstructed = model
            .inverse_transform(&model.transform(&data).unwrap())
            .unwrap();
        let flattened_rebuilt = Array1::from_iter(reconstructed.iter().copied());
        let mean_squared_error =
            sk_mean_squared_error(flattened_truth.view(), flattened_rebuilt.view()).unwrap();
        let determination_score =
            sk_r2_score(flattened_truth.view(), flattened_rebuilt.view()).unwrap();
        assert!(mean_squared_error < previous_mean_squared_error);
        previous_mean_squared_error = mean_squared_error;
        if components == 4 {
            assert!(mean_squared_error < 1e-5);
            assert!(determination_score > 0.9999);
        }
    }
}
