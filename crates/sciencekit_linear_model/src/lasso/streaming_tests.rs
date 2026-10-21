//! Out-of-core `fit_streaming` and incremental `partial_fit` tests for `SKLasso`
//! (spec `lasso`).

use crate::coordinate_descent::test_datasets::{
    SupervisedVecSource, assert_close, chunked_source, large_features, large_targets,
    orthogonal_design, to_csr,
};
use ndarray::{Array1, Array2, array, s};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKSupervisedFit};

use super::core_implementation::SKLasso;

/// A tight lasso over `f64`.
fn lasso(alpha: f64) -> SKLasso<f64> {
    SKLasso::build()
        .alpha(alpha)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
}

/// Uneven batches converge to the dense fit.
#[test]
fn uneven_batches_match_the_dense_fit() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[1, 2, 1]);
    let streamed = lasso(0.5).fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// A single batch matches the dense fit.
#[test]
fn single_batch_matches_the_dense_fit() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[4]);
    let streamed = lasso(0.5).fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// An empty source is rejected with a structured error.
#[test]
fn empty_source_is_rejected() {
    let mut source: SupervisedVecSource<f64> = SupervisedVecSource::new(Vec::new());
    let error = lasso(0.5).fit_streaming(&mut source).unwrap_err();
    assert!(matches!(error, SKError::Conversion(_)));
}

/// Batches with differing column counts are rejected.
#[test]
fn column_mismatch_is_rejected() {
    let batches = vec![
        (array![[1.0, 2.0], [3.0, 4.0]], array![1.0, 2.0]),
        (array![[1.0, 2.0, 3.0]], array![1.0]),
    ];
    let mut source = SupervisedVecSource::new(batches);
    let error = lasso(0.5).fit_streaming(&mut source).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Streaming matches the dense fit at scale.
#[test]
fn streaming_matches_the_dense_fit_on_large_data() {
    let features = large_features::<f64>();
    let targets = large_targets(&features);
    let dense = lasso(0.25).fit(features.view(), targets.view()).unwrap();
    let mut source = chunked_source(&features, &targets, &[3_000, 4_000, 5_000]);
    let streamed = lasso(0.25).fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-5,
    );
    assert_eq!(streamed.row_count(), features.nrows() as u64);
}

/// `partial_fit` accumulates batches to the same fit as one dense fit.
#[test]
fn partial_fit_split_batches_equal_the_whole() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut model = lasso(0.5)
        .fit(features.slice(s![0..1, ..]), targets.slice(s![0..1]))
        .unwrap();
    model
        .partial_fit(features.slice(s![1.., ..]), targets.slice(s![1..]))
        .unwrap();
    assert_close(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
    assert_eq!(model.row_count(), 4);
}

/// An empty incremental batch is a no-op.
#[test]
fn partial_fit_empty_batch_is_a_noop() {
    let (features, targets) = orthogonal_design();
    let mut model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let before = model.coefficients().to_owned();
    let empty_features: Array2<f64> = Array2::zeros((0, 3));
    let empty_targets: Array1<f64> = Array1::zeros(0);
    model
        .partial_fit(empty_features.view(), empty_targets.view())
        .unwrap();
    assert_eq!(model.coefficients(), before.view());
}

/// A column count differing from the fitted design is rejected.
#[test]
fn partial_fit_rejects_a_column_mismatch() {
    let (features, targets) = orthogonal_design();
    let mut model = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let wrong_features: Array2<f64> = Array2::zeros((1, 4));
    let wrong_targets: Array1<f64> = Array1::zeros(1);
    let error = model
        .partial_fit(wrong_features.view(), wrong_targets.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// A sparse incremental batch matches the dense accumulation.
#[test]
fn partial_fit_sparse_batch_matches_dense() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let first = to_csr(&features.slice(s![0..2, ..]).to_owned());
    let rest = to_csr(&features.slice(s![2.., ..]).to_owned());
    let mut model = lasso(0.5)
        .fit(first.view(), targets.slice(s![0..2]))
        .unwrap();
    model
        .partial_fit(rest.view(), targets.slice(s![2..]))
        .unwrap();
    assert_close(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// An in-memory execution intent collects the batches and matches the dense fit.
#[test]
fn in_memory_intent_collects_and_matches_the_dense_fit() {
    use sciencekit_common::SKExecutionMode;

    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let estimator = SKLasso::<f64>::build()
        .alpha(0.5)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[2, 2]);
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// The streaming fit is independent of the batch order.
#[test]
fn streaming_is_independent_of_batch_order() {
    let (features, targets) = orthogonal_design();
    let dense = lasso(0.5).fit(features.view(), targets.view()).unwrap();
    let mut permuted = SupervisedVecSource::new(vec![
        (
            features.slice(s![2.., ..]).to_owned(),
            targets.slice(s![2..]).to_owned(),
        ),
        (
            features.slice(s![0..2, ..]).to_owned(),
            targets.slice(s![0..2]).to_owned(),
        ),
    ]);
    let streamed = lasso(0.5).fit_streaming(&mut permuted).unwrap();
    assert_close(
        &streamed.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}
