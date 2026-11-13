//! Companion tests for `SKIncrementalPCA` repeated-fit determinism,
//! rayon thread-count invariance, batch-size invariance, `Send + Sync` bounds
//! and concurrent `transform` / `inverse_transform`.

use std::sync::Arc;

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKIncrementalPCA, SKIncrementalPCABuilder, SKIncrementalPCAModel};

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
        3.1 * (0.13 * row_scalar + 0.23 * column_scalar).sin()
            + 1.4 * (0.29 * row_scalar - 0.41 * column_scalar).cos()
            + ((row * 5 + column * 7) % 11) as f64 * 0.15
    })
}

#[test]
fn incremental_concurrency_types_are_send_and_sync() {
    assert_send_sync::<SKIncrementalPCA<f64>>();
    assert_send_sync::<SKIncrementalPCA<f32>>();
    assert_send_sync::<SKIncrementalPCABuilder<f64>>();
    assert_send_sync::<SKIncrementalPCABuilder<f32>>();
    assert_send_sync::<SKIncrementalPCAModel<f64>>();
    assert_send_sync::<SKIncrementalPCAModel<f32>>();
}

#[test]
fn incremental_concurrency_repeated_fits_and_thread_counts_are_bit_identical() {
    // 8_192 x 32 = 262_144 elements (> 2 * SK_AXIS_SUM_GRAIN), exercising the
    // parallel 256-row chunk reduction under 4 threads vs sequential under 1 thread.
    let dense = deterministic_design(8_192, 32);
    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(5)
        .batch_size(2_048)
        .whiten(true)
        .build()
        .unwrap();

    let pool_one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool_four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let fit_one_first = pool_one.install(|| estimator.fit(&dense).unwrap());
    let fit_one_second = pool_one.install(|| estimator.fit(&dense).unwrap());
    let fit_four = pool_four.install(|| estimator.fit(&dense).unwrap());

    assert_eq!(fit_one_first.mean(), fit_one_second.mean());
    assert_eq!(
        fit_one_first.scatter_matrix(),
        fit_one_second.scatter_matrix()
    );
    assert_eq!(fit_one_first.components(), fit_one_second.components());
    assert_eq!(fit_one_first.mean(), fit_four.mean());
    assert_eq!(fit_one_first.scatter_matrix(), fit_four.scatter_matrix());
    assert_eq!(fit_one_first.components(), fit_four.components());
    assert_eq!(
        fit_one_first.explained_variance(),
        fit_four.explained_variance()
    );

    let projected_one = pool_one.install(|| fit_one_first.transform(&dense).unwrap());
    let projected_four = pool_four.install(|| fit_four.transform(&dense).unwrap());
    assert_eq!(projected_one, projected_four);

    let rebuilt_one = pool_one.install(|| fit_one_first.inverse_transform(&projected_one).unwrap());
    let rebuilt_four = pool_four.install(|| fit_four.inverse_transform(&projected_four).unwrap());
    assert_eq!(rebuilt_one, rebuilt_four);
}

#[test]
fn incremental_concurrency_batch_size_invariance_holds_for_dense_and_sparse() {
    let dense = Array2::from_shape_fn((95, 6), |(row, column)| {
        if (row * 5 + column * 3) % 4 == 0 {
            let angle = (row as f64 + 1.0) * 0.19 + (column as f64) * 0.51;
            angle.sin() * (8.0 - column as f64) + ((row * 3 + column * 7) % 13) as f64 * 0.25
        } else {
            0.0
        }
    });
    let sparse = csr_from_dense(&dense);

    let reference = SKIncrementalPCA::<f64>::build()
        .number_of_components(4)
        .batch_size(1024)
        .build()
        .unwrap()
        .fit(&dense)
        .unwrap();

    for batch_size in [7usize, 19, 32, 64] {
        let estimator = SKIncrementalPCA::<f64>::build()
            .number_of_components(4)
            .batch_size(batch_size)
            .build()
            .unwrap();
        let dense_variant = estimator.fit(&dense).unwrap();
        let sparse_variant = estimator.fit(sparse.view()).unwrap();

        for (left, right) in dense_variant
            .components()
            .iter()
            .zip(reference.components().iter())
        {
            assert!(
                (left - right).abs() < 1e-9,
                "batch_size={batch_size}: left={left}, right={right}"
            );
        }
        for (left, right) in sparse_variant
            .components()
            .iter()
            .zip(reference.components().iter())
        {
            assert!(
                (left - right).abs() < 1e-9,
                "sparse batch_size={batch_size}: left={left}, right={right}"
            );
        }
    }
}

#[test]
fn incremental_concurrency_shared_model_transforms_match_sequential() {
    let data = deterministic_design(640, 10);
    let sparse_dense = Array2::from_shape_fn((128, 10), |(row, column)| {
        if (row + column * 3) % 4 == 0 {
            (row as f64) * 0.12 - (column as f64) * 0.85
        } else {
            0.0
        }
    });
    let sparse_query = csr_from_dense(&sparse_dense);

    let estimator = Arc::new(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(4)
            .batch_size(128)
            .whiten(true)
            .build()
            .unwrap(),
    );
    let model = Arc::new(estimator.fit(&data).unwrap());
    let expected_dense = model.transform(&data).unwrap();
    let expected_reconstructed = model.inverse_transform(&expected_dense).unwrap();
    let expected_sparse = model.transform(sparse_query.view()).unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..6)
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
