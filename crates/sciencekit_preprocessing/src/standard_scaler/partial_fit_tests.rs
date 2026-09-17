//! Incremental companion tests for the standard scaler: `partial_fit`
//! convergence to the dense result, no-op and mismatch paths, chaining, flag
//! propagation and the large-scale run. Shared datasets live in
//! `test_datasets`.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKError, SKUnsupervisedFit};

use super::test_datasets::{
    assert_close, assert_close_with_tolerance, chunk_rows, default_scaler, large_features,
    small_features,
};
use crate::standard_scaler::core_implementation::SKStandardScaler;

/// Fitting on a prefix then `partial_fit` on the suffix matches the dense fit
/// over the whole data.
#[test]
fn partial_fit_after_fit_matches_dense() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let mut model = default_scaler()
        .fit(&chunk_rows(&features, &[2, 4])[0])
        .expect("fit succeeds");
    model
        .partial_fit(&chunk_rows(&features, &[2, 4])[1])
        .expect("partial fit succeeds");
    assert_close(model.mean(), &reference.mean().to_vec());
    assert_close(model.scale(), &reference.scale().to_vec());
    assert_eq!(model.sample_count(), 6);
}

/// Repeated single-row `partial_fit` calls accumulate to the dense result.
#[test]
fn partial_fit_accumulates_single_rows() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let rows = chunk_rows(&features, &[1, 1, 1, 1, 1, 1]);
    let mut model = default_scaler().fit(&rows[0]).expect("fit succeeds");
    for row in &rows[1..] {
        model.partial_fit(row).expect("partial fit succeeds");
    }
    assert_close(model.mean(), &reference.mean().to_vec());
    assert_close(model.scale(), &reference.scale().to_vec());
    assert_eq!(model.sample_count(), 6);
}

/// An empty batch is a no-op: neither statistics nor count move.
#[test]
fn partial_fit_empty_batch_is_noop() {
    let features = small_features();
    let mut model = default_scaler().fit(&features).expect("fit succeeds");
    let before_mean = model.mean().clone();
    let before_scale = model.scale().clone();
    model
        .partial_fit(&Array2::<f64>::zeros((0, 2)))
        .expect("partial fit succeeds");
    assert_eq!(model.sample_count(), 6);
    assert_eq!(model.mean(), &before_mean);
    assert_eq!(model.scale(), &before_scale);
}

/// A batch with the wrong column count fails without touching the model.
#[test]
fn partial_fit_column_mismatch_errors() {
    let features = small_features();
    let mut model = default_scaler().fit(&features).expect("fit succeeds");
    let error = model
        .partial_fit(&array![[1.0, 2.0, 3.0]])
        .expect_err("mismatched partial fit must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
    assert_eq!(model.sample_count(), 6);
    assert_close(model.mean(), &[6.0, 7.0]);
}

/// `partial_fit` returns the model itself, so updates chain.
#[test]
fn partial_fit_chains_updates() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let chunks = chunk_rows(&features, &[2, 2, 2]);
    let mut model = default_scaler().fit(&chunks[0]).expect("fit succeeds");
    model
        .partial_fit(&chunks[1])
        .expect("partial fit succeeds")
        .partial_fit(&chunks[2])
        .expect("partial fit succeeds");
    assert_close(model.mean(), &reference.mean().to_vec());
    assert_close(model.scale(), &reference.scale().to_vec());
}

/// Disabled flags stay disabled through incremental updates.
#[test]
fn partial_fit_respects_builder_flags() {
    let features = small_features();
    let scaler = SKStandardScaler::build()
        .with_mean(false)
        .with_std(false)
        .build()
        .expect("scaler builds");
    let reference = scaler.fit(&features).expect("fit succeeds");
    let chunks = chunk_rows(&features, &[3, 3]);
    let mut model = scaler.fit(&chunks[0]).expect("fit succeeds");
    model.partial_fit(&chunks[1]).expect("partial fit succeeds");
    assert_close(model.mean(), &reference.mean().to_vec());
    assert_close(model.scale(), &reference.scale().to_vec());
}

/// Incremental updates on the large dataset match the dense fit.
#[test]
fn large_partial_fit_matches_dense_fit() {
    let features = large_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let chunks = chunk_rows(&features, &[30_000, 30_000, 40_000]);
    let mut model = default_scaler().fit(&chunks[0]).expect("fit succeeds");
    model
        .partial_fit(&chunks[1])
        .expect("partial fit succeeds")
        .partial_fit(&chunks[2])
        .expect("partial fit succeeds");
    assert_eq!(model.sample_count(), 100_000);
    assert_close_with_tolerance(model.mean(), &reference.mean().to_vec(), 1e-9);
    assert_close_with_tolerance(model.scale(), &reference.scale().to_vec(), 1e-9);
}
