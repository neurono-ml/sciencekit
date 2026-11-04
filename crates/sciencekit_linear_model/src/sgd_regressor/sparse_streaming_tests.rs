//! Sparse streaming conformance tests for `SKSGDRegressor`.
//!
//! A sparse epoch visits the same rows in the same seeded order as the dense
//! epoch, so a sparse stream equals the dense stream and the dense `fit` under a
//! fixed seed. The guarded test streams a shape whose dense twin is impractical
//! for the test budget; it proves the epoch completes there, not that no dense
//! copy exists.

use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::SKSGDRegressor;
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate};
use crate::sparse_streaming_test_datasets::{
    SparseSupervisedSource, chunk_pairs_csr, guard_matrix, guard_targets,
};
use crate::test_datasets::{assert_close, regression_design};

fn estimator() -> SKSGDRegressor<f64> {
    SKSGDRegressor::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .alpha(0.0)
        .shuffle(true)
        .random_seed(11)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
}

/// The in-memory dense epoch stream helper: one batch covering the whole design.
fn dense_source(
    features: &ndarray::Array2<f64>,
    targets: &ndarray::Array1<f64>,
) -> Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)> {
    vec![(features.clone(), targets.clone())]
}

#[test]
fn sparse_stream_matches_dense_stream_and_fit() {
    let (features, targets) = regression_design();
    let reference = estimator()
        .fit(features.view(), targets.view())
        .expect("dense fit");
    let dense_stream = estimator()
        .fit_streaming(&mut crate::test_datasets::SKSupervisedFloatSource::new(
            dense_source(&features, &targets),
        ))
        .expect("dense stream");
    let sparse_stream = estimator()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[5],
        )))
        .expect("sparse stream");
    for model in [&reference, &dense_stream] {
        assert_close(
            model.coefficients().as_slice().unwrap(),
            sparse_stream.coefficients().as_slice().unwrap(),
            1e-9,
        );
        assert_close(&[model.intercept()], &[sparse_stream.intercept()], 1e-9);
    }
    assert_eq!(sparse_stream.epochs_taken(), reference.epochs_taken());
}

/// A sparse epoch completes at a shape whose dense twin is impractical for the
/// test budget.
#[test]
fn sparse_stream_handles_a_shape_beyond_the_test_budget() {
    let rows = 200_000;
    let mut source =
        SparseSupervisedSource::<f64>::new(vec![(guard_matrix(rows, 64), guard_targets(rows))]);
    let streamed = SKSGDRegressor::<f64>::build()
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.coefficients().len(), 64);
    assert_eq!(streamed.epochs_taken(), 1);
}
