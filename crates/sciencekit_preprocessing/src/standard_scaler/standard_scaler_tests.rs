//! Companion tests for the standard scaler: executable truth quoted by the
//! Development Explained book. Every test drives the public API only
//! (`build` → `fit` → `transform`), never private fields. Shared datasets and
//! helpers live in `test_datasets`.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKFeatureTransformer, SKFloat, SKUnsupervisedFit,
};

use super::test_datasets::{
    assert_close, chunk_rows, default_scaler, large_features, small_features,
};
use crate::standard_scaler::core_implementation::SKStandardScaler;

/// Fit learns per-column mean and population scale.
#[test]
fn fit_learns_per_column_mean_and_scale() {
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.mean(), &[2.0, 3.0]);
    assert_close(model.scale(), &[1.0, 1.0]);
}

/// Transform centers and scales to z-scores.
#[test]
fn transform_produces_z_scores() {
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0, -1.0], [1.0, 1.0]]);
}

/// `with_mean = false` learns zero means and leaves values uncentered.
#[test]
fn without_mean_keeps_values_uncentered() {
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let scaler = SKStandardScaler::build()
        .with_mean(false)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_close(model.mean(), &[0.0, 0.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, features);
}

/// `with_std = false` learns unit scales and only centers.
#[test]
fn without_std_centers_without_scaling() {
    let features: Array2<f64> = array![[0.0, 0.0], [2.0, 4.0]];
    let scaler = SKStandardScaler::build()
        .with_std(false)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_close(model.mean(), &[1.0, 2.0]);
    assert_close(model.scale(), &[1.0, 1.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0, -2.0], [1.0, 2.0]]);
}

/// A constant column has zero variance: its scale is one and its transformed
/// values are zero instead of NaN.
#[test]
fn constant_column_yields_unit_scale_and_zero_output() {
    let features: Array2<f64> = array![[5.0, 1.0], [5.0, 3.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.scale(), &[1.0, 1.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[0.0, -1.0], [0.0, 1.0]]);
}

/// The estimator stays generic over both supported floats.
#[test]
fn fit_accepts_either_float() {
    fn mean_of<F: SKFloat>(raw: Vec<F>) -> Array1<F> {
        let features = Array2::from_shape_vec((2, 1), raw).unwrap();
        default_scaler()
            .fit(&features)
            .expect("fit succeeds")
            .mean()
            .clone()
    }
    assert_eq!(mean_of(vec![1.0_f32, 3.0]), array![2.0_f32]);
    assert_eq!(mean_of(vec![1.0_f64, 3.0]), array![2.0_f64]);
}

/// Empty inputs fail with a shape error before any element is processed.
#[test]
fn fit_rejects_empty_input() {
    let features: Array2<f64> = Array2::zeros((0, 2));
    let error = default_scaler()
        .fit(&features)
        .expect_err("empty fit must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Sparse inputs are rejected at the dense-only boundary with a precise error.
#[test]
fn fit_rejects_sparse_input() {
    let sparse = sprs::CsMat::new((2, 2), vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]);
    let error = default_scaler()
        .fit(sparse.view())
        .expect_err("sparse fit must fail");
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

/// Transforming data with a different column count than the fitted model fails.
#[test]
fn transform_rejects_column_count_mismatch() {
    let fitted: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let other: Array2<f64> = array![[1.0, 2.0, 3.0]];
    let model = default_scaler().fit(&fitted).expect("fit succeeds");
    let error = model
        .transform(&other)
        .expect_err("mismatched transform must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The default backend is Faer on both estimator and model.
#[test]
fn backend_defaults_to_faer() {
    let scaler = default_scaler();
    assert_eq!(scaler.backend(), SKBackendKind::Faer);
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::Faer);
}

/// A backend override flows from builder into estimator and fitted model.
#[test]
fn backend_override_reaches_model() {
    let scaler = SKStandardScaler::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("scaler builds");
    assert_eq!(scaler.backend(), SKBackendKind::MatrixMultiply);
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0, -1.0], [1.0, 1.0]]);
}

/// The large dense fit produces finite statistics with the constant column at
/// unit scale.
#[test]
fn large_dense_fit_learns_finite_statistics() {
    let features = large_features();
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_eq!(model.mean().len(), 8);
    assert!(model.mean().iter().all(|value| value.is_finite()));
    assert!(model.scale().iter().all(|value| value.is_finite()));
    assert!(model.scale().iter().take(7).all(|&value| value > 0.0));
    assert_eq!(model.scale()[7], 1.0);
}

/// Row chunking covers every row exactly once, in order.
#[test]
fn chunk_rows_covers_every_row_once() {
    let features = small_features();
    let chunks = chunk_rows(&features, &[2, 1, 3]);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].nrows(), 2);
    assert_eq!(chunks[2].nrows(), 3);
    assert_eq!(chunks[1][[0, 1]], 6.0);
}
