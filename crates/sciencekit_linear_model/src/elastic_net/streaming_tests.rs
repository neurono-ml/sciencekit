//! Out-of-core `fit_streaming` and incremental `partial_fit` tests for
//! `SKElasticNet` (spec `elastic-net`).

use crate::coordinate_descent::test_datasets::{
    SupervisedVecSource, assert_close, chunked_source, orthogonal_design, to_csr,
};
use ndarray::{Array1, Array2, array, s};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKSupervisedFit};

use super::core_implementation::SKElasticNet;

/// A tight elastic net over `f64`.
fn elastic_net(alpha: f64, l1_ratio: f64) -> SKElasticNet<f64> {
    SKElasticNet::build()
        .alpha(alpha)
        .l1_ratio(l1_ratio)
        .tolerance(1e-10)
        .maximum_number_of_iterations(100_000)
        .build()
        .unwrap()
}

/// Uneven batches converge to the dense fit.
#[test]
fn uneven_batches_match_the_dense_fit() {
    let (features, targets) = orthogonal_design();
    let dense = elastic_net(0.5, 0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    let mut source = chunked_source(&features, &targets, &[1, 2, 1]);
    let streamed = elastic_net(0.5, 0.5).fit_streaming(&mut source).unwrap();
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
    let error = elastic_net(0.5, 0.5).fit_streaming(&mut source).unwrap_err();
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
    let error = elastic_net(0.5, 0.5).fit_streaming(&mut source).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// `partial_fit` accumulates batches to the same fit as one dense fit.
#[test]
fn partial_fit_split_batches_equal_the_whole() {
    let (features, targets) = orthogonal_design();
    let dense = elastic_net(0.5, 0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    let mut model = elastic_net(0.5, 0.5)
        .fit(features.slice(s![0..2, ..]), targets.slice(s![0..2]))
        .unwrap();
    model
        .partial_fit(features.slice(s![2.., ..]), targets.slice(s![2..]))
        .unwrap();
    assert_close(
        &model.coefficient_for(0).to_vec(),
        &dense.coefficient_for(0).to_vec(),
        1e-8,
    );
}

/// An empty incremental batch is a no-op.
#[test]
fn partial_fit_empty_batch_is_a_noop() {
    let (features, targets) = orthogonal_design();
    let mut model = elastic_net(0.5, 0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    let before = model.coefficients().to_owned();
    let empty_features: Array2<f64> = Array2::zeros((0, 3));
    let empty_targets: Array1<f64> = Array1::zeros(0);
    model
        .partial_fit(empty_features.view(), empty_targets.view())
        .unwrap();
    assert_eq!(model.coefficients(), before.view());
}

/// A sparse incremental batch matches the dense accumulation.
#[test]
fn partial_fit_sparse_batch_matches_dense() {
    let (features, targets) = orthogonal_design();
    let dense = elastic_net(0.5, 0.5)
        .fit(features.view(), targets.view())
        .unwrap();
    let first = to_csr(&features.slice(s![0..2, ..]).to_owned());
    let rest = to_csr(&features.slice(s![2.., ..]).to_owned());
    let mut model = elastic_net(0.5, 0.5)
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
