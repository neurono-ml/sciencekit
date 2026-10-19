//! Companion tests for the min/max scaler: executable truth quoted by the
//! Developer Book. Every test drives the public API only
//! (`build` → `fit` → `transform`), never private fields. Shared datasets and
//! helpers live in `test_datasets`.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat,
    SKUnsupervisedFit,
};

use super::test_datasets::{assert_close, default_scaler, large_features};
use crate::minmax_scaler::core_implementation::SKMinMaxScaler;

/// The defaults map features onto the unit range, without clipping, carrying
/// `Faer` and an `Automatic` intent.
#[test]
fn default_configuration() {
    let scaler = default_scaler();
    assert_eq!(scaler.feature_range(), (0.0, 1.0));
    assert!(!scaler.clip());
    assert_eq!(scaler.backend(), SKBackendKind::Faer);
    assert_eq!(scaler.execution_intent(), SKExecutionMode::Automatic);
}

/// A column `[0, 10]` with the default range maps min → 0, max → 1 and the
/// midpoint → 0.5.
#[test]
fn fit_maps_column_to_unit_range() {
    let features: Array2<f64> = array![[0.0], [5.0], [10.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.minimum(), &[0.0]);
    assert_close(model.range(), &[10.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[0.0], [0.5], [1.0]]);
}

/// The learned minimum and range are exposed per column.
#[test]
fn fit_learns_per_column_minimum_and_range() {
    let features: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 12.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.minimum(), &[1.0, 2.0]);
    assert_close(model.range(), &[4.0, 10.0]);
    assert_close(model.scale(), &[0.25, 0.1]);
}

/// A `feature_range` override maps the data into the configured target range.
#[test]
fn feature_range_override_maps_into_target_range() {
    let features: Array2<f64> = array![[0.0], [5.0], [10.0]];
    let scaler = SKMinMaxScaler::<f64>::build()
        .feature_range((-1.0, 1.0))
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0], [0.0], [1.0]]);
}

/// A constant column has zero range, unit scale and maps every value to the
/// target-range minimum rather than `NaN`.
#[test]
fn constant_column_maps_to_target_minimum() {
    let features: Array2<f64> = array![[5.0, 1.0], [5.0, 3.0], [5.0, 5.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.range(), &[0.0, 4.0]);
    assert_close(model.scale(), &[1.0, 0.25]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]);
}

/// With `clip = true`, values beyond the fitted range clamp to the target range.
#[test]
fn clip_true_constrains_output_to_feature_range() {
    let train: Array2<f64> = array![[0.0], [10.0]];
    let scaler = SKMinMaxScaler::<f64>::build()
        .clip(true)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&train).expect("fit succeeds");
    assert!(model.clip());
    let output = model
        .transform(&array![[15.0], [-5.0]])
        .expect("transform succeeds");
    assert_eq!(output, array![[1.0], [0.0]]);
}

/// With the default `clip = false`, out-of-range values fall outside the target
/// range.
#[test]
fn clip_false_allows_out_of_range_output() {
    let train: Array2<f64> = array![[0.0], [10.0]];
    let model = default_scaler().fit(&train).expect("fit succeeds");
    let output = model
        .transform(&array![[15.0]])
        .expect("transform succeeds");
    assert_eq!(output, array![[1.5]]);
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

/// Transforming data with a different column count than the fitted model fails
/// and produces no output.
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

/// The default backend is `Faer` on both estimator and model.
#[test]
fn backend_defaults_to_faer() {
    let scaler = default_scaler();
    assert_eq!(scaler.backend(), SKBackendKind::Faer);
    let features: Array2<f64> = array![[1.0], [2.0]];
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::Faer);
}

/// A backend override flows from builder into estimator and fitted model.
#[test]
fn backend_override_reaches_model() {
    let scaler = SKMinMaxScaler::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("scaler builds");
    assert_eq!(scaler.backend(), SKBackendKind::MatrixMultiply);
    let features: Array2<f64> = array![[1.0], [2.0]];
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[0.0], [1.0]]);
}

/// The estimator stays generic over both supported floats.
#[test]
fn fit_accepts_either_float() {
    fn minimum_of<F: SKFloat>(raw: Vec<F>) -> F {
        let features = Array2::from_shape_vec((2, 1), raw).unwrap();
        SKMinMaxScaler::<F>::build()
            .build()
            .expect("scaler builds")
            .fit(&features)
            .expect("fit succeeds")
            .minimum()[0]
    }
    assert_eq!(minimum_of(vec![1.0_f32, 3.0]), 1.0_f32);
    assert_eq!(minimum_of(vec![1.0_f64, 3.0]), 1.0_f64);
}

/// The large dense fit learns finite parameters with the constant column at
/// zero range.
#[test]
fn large_dense_fit_learns_finite_statistics() {
    let features = large_features();
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_eq!(model.minimum().len(), 8);
    assert!(model.minimum().iter().all(|value| value.is_finite()));
    assert!(model.range().iter().all(|value| value.is_finite()));
    assert!(model.range().iter().take(7).all(|&value| value > 0.0));
    assert_eq!(model.range()[7], 0.0);
}

/// `assert_close` accepts an `Array1` produced by `to_vec`-style access.
#[test]
fn close_helper_matches_exact_vectors() {
    let values: Array1<f64> = array![1.0, 2.0, 3.0];
    assert_close(&values, &[1.0, 2.0, 3.0]);
}

/// The immutable estimator and the fitted model are safe to share across
/// threads: concurrent fits and transforms agree with the single-threaded
/// result, on the large dataset.
#[test]
fn estimator_and_model_are_safe_under_concurrency() {
    use std::sync::Arc;

    let features: Arc<Array2<f64>> = Arc::new(large_features());
    let scaler = Arc::new(default_scaler());
    let model = Arc::new(scaler.fit(&*features).expect("fit succeeds"));
    let expected = model.transform(&*features).expect("transform succeeds");

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let scaler = Arc::clone(&scaler);
            let model = Arc::clone(&model);
            let features = Arc::clone(&features);
            std::thread::spawn(move || {
                let fitted = scaler.fit(&*features).expect("concurrent fit succeeds");
                assert_eq!(fitted.minimum(), model.minimum());
                assert_eq!(fitted.range(), model.range());
                model
                    .transform(&*features)
                    .expect("concurrent transform succeeds")
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().expect("worker thread joins"), expected);
    }
}
