//! Companion tests for `SKTruncatedSVD` determinism, Rayon thread-count
//! invariance and concurrent sharing (`Send + Sync`).

use std::sync::Arc;

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKExecutionMode, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKTruncatedSVD, SKTruncatedSVDAlgorithm, SKTruncatedSVDBuilder, SKTruncatedSVDModel};

fn assert_send_sync<T: Send + Sync>() {}

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

fn deterministic_design(rows: usize, columns: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, columns), |(row, column)| {
        let row_scalar = row as f64;
        let column_scalar = (column + 1) as f64;
        2.6 * (0.13 * row_scalar + 0.2 * column_scalar).sin()
            + 1.2 * (0.29 * row_scalar - 0.4 * column_scalar).cos()
            + 0.7 * column_scalar
    })
}

fn deterministic_wide_csr(rows: usize, columns: usize) -> CsMat<f64> {
    let mut triplet = TriMat::with_capacity((rows, columns), rows * 4);
    (0..rows).for_each(|row| {
        triplet.add_triplet(row, row % columns, (row as f64 + 1.0) * 0.75);
        triplet.add_triplet(row, (row * 5 + 3) % columns, -((row % 9) as f64 + 1.25));
        triplet.add_triplet(
            row,
            (row * 11 + 7) % columns,
            ((row % 4) as f64 + 0.5) * 0.4,
        );
    });
    triplet.to_csr()
}

#[test]
fn truncated_concurrency_estimator_builder_and_model_are_send_and_sync() {
    assert_send_sync::<SKTruncatedSVD<f64>>();
    assert_send_sync::<SKTruncatedSVD<f32>>();
    assert_send_sync::<SKTruncatedSVDBuilder<f64>>();
    assert_send_sync::<SKTruncatedSVDModel<f64>>();
    assert_send_sync::<SKTruncatedSVDModel<f32>>();
    assert_send_sync::<SKTruncatedSVDAlgorithm>();
}

#[test]
fn truncated_concurrency_repeated_fits_agree_across_runs_and_rayon_thread_counts() {
    // 4_096 x 32 = 131_072 elements (> SK_AXIS_SUM_GRAIN = 100_000), exercising
    // multi-threaded Rayon chunks under pool_four and sequential under pool_one.
    let dense = deterministic_design(4_096, 32);
    let sparse = deterministic_wide_csr(2_048, 64);

    let pool_one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool_four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(3)
        .random_seed(29)
        .execution_mode(SKExecutionMode::InProcessAsynchronous)
        .build()
        .unwrap();

    // Repeated fits on the same thread pool agree bit-for-bit.
    let repeated_a = estimator.fit(&dense).unwrap();
    let repeated_b = estimator.fit(&dense).unwrap();
    assert_eq!(repeated_a, repeated_b);
    assert_eq!(
        repeated_a.transform(&dense).unwrap(),
        repeated_b.transform(&dense).unwrap()
    );

    // 1-thread and 4-thread Rayon pools produce identical models and projections.
    let dense_one = pool_one.install(|| estimator.fit(&dense).unwrap());
    let dense_four = pool_four.install(|| estimator.fit(&dense).unwrap());
    assert_eq!(dense_one, dense_four);
    let projected_one = pool_one.install(|| dense_one.transform(&dense).unwrap());
    let projected_four = pool_four.install(|| dense_four.transform(&dense).unwrap());
    assert_eq!(projected_one, projected_four);
    assert_eq!(
        pool_one.install(|| dense_one.inverse_transform(&projected_one).unwrap()),
        pool_four.install(|| dense_four.inverse_transform(&projected_four).unwrap())
    );

    let sparse_one = pool_one.install(|| estimator.fit(sparse.view()).unwrap());
    let sparse_four = pool_four.install(|| estimator.fit(sparse.view()).unwrap());
    assert_eq!(sparse_one.components(), sparse_four.components());
    assert_eq!(
        pool_one.install(|| sparse_one.transform(sparse.view()).unwrap()),
        pool_four.install(|| sparse_four.transform(sparse.view()).unwrap())
    );
}

#[test]
fn truncated_concurrency_shared_model_transforms_match_sequential_output() {
    let data = deterministic_design(600, 14);
    let sparse_query_dense = Array2::from_shape_fn((300, 14), |(row, column)| {
        if (row + column) % 4 == 0 {
            (row as f64) * 0.15 - (column as f64)
        } else {
            0.0
        }
    });
    let sparse_query = csr_from_dense(&sparse_query_dense);

    let estimator = Arc::new(
        SKTruncatedSVD::<f64>::build()
            .number_of_components(4)
            .random_seed(91)
            .build()
            .unwrap(),
    );
    let model = Arc::new(estimator.fit(&data).unwrap());
    let expected_dense = model.transform(&data).unwrap();
    let expected_reconstructed = model.inverse_transform(&expected_dense).unwrap();
    let expected_sparse = model.transform(sparse_query.view()).unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|worker| {
                let model = Arc::clone(&model);
                let estimator = Arc::clone(&estimator);
                let data_ref = &data;
                let sparse_ref = &sparse_query;
                scope.spawn(move || {
                    if worker % 2 == 0 {
                        let refitted = estimator.fit(data_ref).unwrap();
                        assert_eq!(refitted.components(), model.components());
                    }
                    let projected = model.transform(data_ref).unwrap();
                    let rebuilt = model.inverse_transform(&projected).unwrap();
                    let projected_sparse = model.transform(sparse_ref.view()).unwrap();
                    (projected, rebuilt, projected_sparse)
                })
            })
            .collect();

        for handle in handles {
            let (projected, rebuilt, projected_sparse) = handle.join().unwrap();
            assert_eq!(projected, expected_dense);
            assert_eq!(rebuilt, expected_reconstructed);
            assert_eq!(projected_sparse, expected_sparse);
        }
    });
}
