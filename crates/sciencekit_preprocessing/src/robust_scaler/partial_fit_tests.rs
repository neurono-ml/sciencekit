//! Incremental companion tests for the robust scaler: the online quantile
//! sketch approximating the exact dense fit, the no-op and mismatch paths and
//! sample-count accumulation. Shared datasets live in `test_datasets`.

use ndarray::array;
use sciencekit_common::{SKError, SKUnsupervisedFit};

use super::test_datasets::{
    SKETCH_RELATIVE_TOLERANCE, assert_close, assert_close_with_tolerance, chunk_rows,
    default_scaler, large_features, small_features,
};

/// Fitting on a prefix then `partial_fit` on the rest approximates the exact
/// dense fit within the documented tolerance.
#[test]
fn partial_fit_after_fit_approximates_dense() {
    let features = large_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let chunks = chunk_rows(&features, &[30_000, 30_000, 40_000]);
    let mut model = default_scaler().fit(&chunks[0]).expect("fit succeeds");
    model.partial_fit(&chunks[1]).expect("partial fit succeeds");
    model.partial_fit(&chunks[2]).expect("partial fit succeeds");
    assert_eq!(model.sample_count(), 100_000);
    assert_close_with_tolerance(
        model.center(),
        &reference.center().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
    assert_close_with_tolerance(
        model.scale(),
        &reference.scale().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
}

/// The sample count accumulates across `fit` and `partial_fit`.
#[test]
fn partial_fit_accumulates_sample_count() {
    let features = small_features();
    let chunks = chunk_rows(&features, &[2, 3]);
    let mut model = default_scaler().fit(&chunks[0]).expect("fit succeeds");
    assert_eq!(model.sample_count(), 2);
    model.partial_fit(&chunks[1]).expect("partial fit succeeds");
    assert_eq!(model.sample_count(), 5);
}

/// An empty batch is a no-op: neither statistics nor count move.
#[test]
fn partial_fit_empty_batch_is_noop() {
    let features = small_features();
    let mut model = default_scaler().fit(&features).expect("fit succeeds");
    let before_center = model.center().clone();
    let before_scale = model.scale().clone();
    model
        .partial_fit(&ndarray::Array2::<f64>::zeros((0, 1)))
        .expect("partial fit succeeds");
    assert_eq!(model.sample_count(), 5);
    assert_eq!(model.center(), &before_center);
    assert_eq!(model.scale(), &before_scale);
}

/// A batch with the wrong column count fails without touching the model.
#[test]
fn partial_fit_column_mismatch_errors() {
    let features = small_features();
    let mut model = default_scaler().fit(&features).expect("fit succeeds");
    let error = model
        .partial_fit(&array![[1.0, 2.0]])
        .expect_err("mismatched partial fit must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
    assert_eq!(model.sample_count(), 5);
    assert_close(model.center(), &[3.0]);
}
