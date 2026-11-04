//! Sparse streaming conformance tests for the coordinate-descent models.
//!
//! `SKLasso` and `SKElasticNet` retain the stacked CSR batches and re-solve, so
//! a sparse stream equals the dense stream and the dense `fit`. The guarded test
//! retains a CSR design at a shape whose dense twin is impractical for the test
//! budget.

use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::test_datasets::{assert_close, chunked_source, orthogonal_design};
use crate::elastic_net::SKElasticNet;
use crate::lasso::SKLasso;
use crate::sparse_streaming_test_datasets::{
    SparseSupervisedSource, chunk_pairs_csr, guard_matrix, guard_targets,
};

#[test]
fn lasso_sparse_stream_matches_dense_stream_and_fit() {
    let (features, targets) = orthogonal_design();
    let reference = SKLasso::<f64>::build()
        .alpha(0.5)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let dense_stream = SKLasso::<f64>::build()
        .alpha(0.5)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
        .fit_streaming(&mut chunked_source(&features, &targets, &[1, 2, 1]))
        .unwrap();
    let sparse_stream = SKLasso::<f64>::build()
        .alpha(0.5)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[1, 2, 1],
        )))
        .unwrap();
    for model in [&reference, &dense_stream] {
        assert_close(
            &sparse_stream.coefficient_for(0).to_vec(),
            &model.coefficient_for(0).to_vec(),
            1e-8,
        );
    }
}

#[test]
fn elastic_net_sparse_stream_matches_dense_stream_and_fit() {
    let (features, targets) = orthogonal_design();
    let build = || {
        SKElasticNet::<f64>::build()
            .alpha(0.5)
            .l1_ratio(0.5)
            .tolerance(1e-10)
            .maximum_number_of_iterations(100_000)
            .build()
            .unwrap()
    };
    let reference = build().fit(features.view(), targets.view()).unwrap();
    let dense_stream = build()
        .fit_streaming(&mut chunked_source(&features, &targets, &[1, 2, 1]))
        .unwrap();
    let sparse_stream = build()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[1, 2, 1],
        )))
        .unwrap();
    for model in [&reference, &dense_stream] {
        assert_close(
            &sparse_stream.coefficient_for(0).to_vec(),
            &model.coefficient_for(0).to_vec(),
            1e-8,
        );
    }
}

/// The retained design streams at a shape whose dense twin is impractical for
/// the test budget.
#[test]
fn sparse_stream_handles_a_shape_beyond_the_test_budget() {
    let rows = 200_000;
    let mut source =
        SparseSupervisedSource::<f64>::new(vec![(guard_matrix(rows, 32), guard_targets(rows))]);
    let streamed = SKLasso::<f64>::build()
        .alpha(0.1)
        .maximum_number_of_iterations(20)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.coefficient_for(0).len(), 32);
    assert_eq!(streamed.row_count(), rows as u64);
}
