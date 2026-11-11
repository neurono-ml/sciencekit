//! Companion tests for PCA determinism, thread-count invariance and concurrent
//! sharing (`Send + Sync`).

use std::sync::Arc;

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKPCA, SKPCABuilder, SKPCAModel, SKPCASolver};

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
        let r = row as f64;
        let c = (column + 1) as f64;
        2.8 * (0.13 * r + 0.2 * c).sin() + 1.1 * (0.29 * r - 0.4 * c).cos() + 0.5 * c
    })
}

#[test]
fn pca_concurrency_estimator_builder_and_model_are_send_and_sync() {
    assert_send_sync::<SKPCA<f64>>();
    assert_send_sync::<SKPCA<f32>>();
    assert_send_sync::<SKPCABuilder<f64>>();
    assert_send_sync::<SKPCAModel<f64>>();
    assert_send_sync::<SKPCAModel<f32>>();
}

#[test]
fn pca_concurrency_repeated_fits_agree_across_runs_and_rayon_thread_counts() {
    let dense = deterministic_design(64, 10);
    let mut sparse_source = Array2::<f64>::zeros((64, 12));
    for row in 0..64 {
        sparse_source[(row, row % 12)] = (row as f64 + 1.0) * 0.75;
        sparse_source[(row, (row * 5 + 3) % 12)] = -((row % 7) as f64 + 1.25);
    }
    let sparse = csr_from_dense(&sparse_source);

    for solver in [
        SKPCASolver::Full,
        SKPCASolver::Randomized,
        SKPCASolver::CovarianceEigh,
    ] {
        let estimator = SKPCA::<f64>::build()
            .number_of_components(3)
            .svd_solver(solver)
            .random_seed(29)
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

        let first = pool_one.install(|| estimator.fit(&dense).unwrap());
        let second = pool_four.install(|| estimator.fit(&dense).unwrap());
        assert_eq!(first.components(), second.components());
        assert_eq!(first.singular_values(), second.singular_values());
        assert_eq!(first.explained_variance(), second.explained_variance());
        assert_eq!(
            pool_one.install(|| first.transform(&dense).unwrap()),
            pool_four.install(|| second.transform(&dense).unwrap())
        );
    }

    let sparse_estimator = SKPCA::<f64>::build()
        .number_of_components(3)
        .svd_solver(SKPCASolver::Randomized)
        .random_seed(29)
        .build()
        .unwrap();
    let sparse_first = sparse_estimator.fit(sparse.view()).unwrap();
    let sparse_second = sparse_estimator.fit(sparse.view()).unwrap();
    assert_eq!(sparse_first.components(), sparse_second.components());
    assert_eq!(
        sparse_first.transform(sparse.view()).unwrap(),
        sparse_second.transform(sparse.view()).unwrap()
    );
}

#[test]
fn pca_concurrency_shared_model_transforms_match_sequential_output() {
    let data = deterministic_design(600, 12);
    let sparse_query_dense = Array2::from_shape_fn((120, 12), |(row, column)| {
        if (row + column) % 4 == 0 {
            (row as f64) * 0.1 - (column as f64)
        } else {
            0.0
        }
    });
    let sparse_query = csr_from_dense(&sparse_query_dense);

    let estimator = Arc::new(
        SKPCA::<f64>::build()
            .number_of_components(4)
            .whiten(true)
            .random_seed(91)
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
