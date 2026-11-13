//! Sparse streaming conformance tests for `SKSGDClassifier`.
//!
//! A sparse epoch visits the same rows in the same seeded order as the dense
//! epoch, so a sparse stream equals the dense stream and the dense `fit` under a
//! fixed seed. The guarded test streams a shape whose dense twin is impractical
//! for the test budget; it proves the epoch completes there, not that no dense
//! copy exists.

use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::SKSGDClassifier;
use crate::sgd::SKConvergencePolicy;
use crate::sparse_streaming_test_datasets::{
    SparseSupervisedSource, chunk_pairs_csr, guard_matrix,
};
use crate::test_datasets::{
    SKSupervisedFloatSource, assert_close, labels_i64, labels_scalar, separable_binary_design,
};

fn estimator() -> SKSGDClassifier<f64> {
    SKSGDClassifier::<f64>::build()
        .shuffle(true)
        .random_seed(7)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
}

#[test]
fn sparse_stream_matches_dense_stream_and_fit() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let reference = estimator()
        .fit(features.view(), labels.view())
        .expect("dense fit");
    let targets = labels_scalar::<f64>(&raw);
    let dense_stream = estimator()
        .fit_streaming(&mut SKSupervisedFloatSource::new(vec![(
            features.clone(),
            targets.clone(),
        )]))
        .expect("dense stream");
    let sparse_stream = estimator()
        .fit_streaming(&mut SparseSupervisedSource::new(chunk_pairs_csr(
            &features,
            &targets,
            &[6],
        )))
        .expect("sparse stream");
    for model in [&reference, &dense_stream] {
        assert_close(
            model.coefficients().as_slice().unwrap(),
            sparse_stream.coefficients().as_slice().unwrap(),
            1e-9,
        );
        assert_close(
            model.intercepts().as_slice().unwrap(),
            sparse_stream.intercepts().as_slice().unwrap(),
            1e-9,
        );
    }
}

/// A sparse epoch completes at a shape whose dense twin is impractical for the
/// test budget.
#[test]
fn sparse_stream_handles_a_shape_beyond_the_test_budget() {
    let rows = 200_000;
    let labels = ndarray::Array1::from_shape_fn(rows, |row| (row % 2) as f64);
    let mut source = SparseSupervisedSource::<f64>::new(vec![(guard_matrix(rows, 64), labels)]);
    let streamed = SKSGDClassifier::<f64>::build()
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.coefficients().ncols(), 64);
    assert_eq!(streamed.epochs_taken(), 1);
}

/// A reopenable sparse provider feeds several epochs, carrying the schedule.
#[test]
fn sparse_reopenable_provider_runs_multiple_epochs() {
    let (features, labels) = separable_binary_design();
    let labels = ndarray::Array1::from_iter(labels.into_iter().map(|label| label as f64));
    let batches = chunk_pairs_csr(&features, &labels, &[6]);
    let provider = sciencekit_common::SKReopenableSource::new(move || {
        Ok(SparseSupervisedSource::<f64>::new(batches.clone()))
    });
    let streamed = SKSGDClassifier::<f64>::build()
        .maximum_number_of_iterations(3)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit_streaming(provider)
        .unwrap();
    assert_eq!(streamed.epochs_taken(), 3);
    assert_eq!(streamed.coefficients().ncols(), features.ncols());
}
