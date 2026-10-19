//! Companion tests for the robust scaler: executable truth quoted by the
//! Developer Book. Every test drives the public API only
//! (`build` → `fit` → `transform`), never private fields.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat,
    SKUnsupervisedFit,
};

use super::test_datasets::{assert_close, assert_close_with_tolerance, default_scaler, large_features};
use crate::robust_scaler::core_implementation::SKRobustScaler;

/// The defaults center on the median and scale by the 25–75 interquartile
/// range, without unit-variance adjustment, carrying `Faer` and an `Automatic`
/// intent.
#[test]
fn default_configuration() {
    let scaler = default_scaler();
    assert!(scaler.with_centering());
    assert!(scaler.with_scaling());
    assert_eq!(scaler.quantile_range(), (25.0, 75.0));
    assert!(!scaler.unit_variance());
    assert_eq!(scaler.backend(), SKBackendKind::Faer);
    assert_eq!(scaler.execution_intent(), SKExecutionMode::Automatic);
}

/// A column `[1, 2, 3, 4, 5]` fitted with `(25, 75)` has center `3` and scale
/// `2` (the 75th minus the 25th percentile).
#[test]
fn fit_learns_median_and_interquartile_range() {
    let features: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.center(), &[3.0]);
    assert_close(model.scale(), &[2.0]);
}

/// Transform centers and scales: the median maps to 0 and the interquartile
/// range maps to 1.
#[test]
fn transform_centers_and_scales() {
    let features: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-1.0], [-0.5], [0.0], [0.5], [1.0]]);
}

/// A `quantile_range` override changes the spread the scale is built from.
#[test]
fn quantile_range_override_changes_scale() {
    let features: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let scaler = SKRobustScaler::<f64>::build()
        .quantile_range((0.0, 100.0))
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_close(model.center(), &[3.0]);
    assert_close(model.scale(), &[4.0]);
}

/// `with_centering = false` learns a zero center and leaves values uncentered.
#[test]
fn without_centering_keeps_values_uncentered() {
    let features: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let scaler = SKRobustScaler::<f64>::build()
        .with_centering(false)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_close(model.center(), &[0.0]);
    assert_close(model.scale(), &[2.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[0.5], [1.0], [1.5], [2.0], [2.5]]);
}

/// `with_scaling = false` learns a unit scale and only centers.
#[test]
fn without_scaling_centers_without_scaling() {
    let features: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let scaler = SKRobustScaler::<f64>::build()
        .with_scaling(false)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_close(model.center(), &[3.0]);
    assert_close(model.scale(), &[1.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert_eq!(output, array![[-2.0], [-1.0], [0.0], [1.0], [2.0]]);
}

/// A zero-spread column gets a unit scale, so its output stays finite.
#[test]
fn zero_spread_column_gets_unit_scale_and_finite_output() {
    let features: Array2<f64> = array![[5.0], [5.0], [5.0]];
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_close(model.scale(), &[1.0]);
    let output = model.transform(&features).expect("transform succeeds");
    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(output, array![[0.0], [0.0], [0.0]]);
}

/// With `unit_variance`, the scale is divided by the standard-normal 25–75 span
/// (approximately `1.34898`).
#[test]
fn unit_variance_divides_scale_by_standard_normal_span() {
    let features: Array2<f64> = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let scaler = SKRobustScaler::<f64>::build()
        .unit_variance(true)
        .build()
        .expect("scaler builds");
    let model = scaler.fit(&features).expect("fit succeeds");
    assert!(model.unit_variance());
    assert_close_with_tolerance(
        model.scale(),
        &[2.0 / 1.348_979_500_392_163_4],
        1e-9,
    );
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
    let scaler = SKRobustScaler::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .expect("scaler builds");
    assert_eq!(scaler.backend(), SKBackendKind::MatrixMultiply);
    let features: Array2<f64> = array![[1.0], [2.0]];
    let model = scaler.fit(&features).expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}

/// The estimator stays generic over both supported floats.
#[test]
fn fit_accepts_either_float() {
    fn center_of<F: SKFloat>(raw: Vec<F>) -> F {
        let features = Array2::from_shape_vec((3, 1), raw).unwrap();
        SKRobustScaler::<F>::build()
            .build()
            .expect("scaler builds")
            .fit(&features)
            .expect("fit succeeds")
            .center()[0]
    }
    assert_eq!(center_of(vec![1.0_f32, 2.0, 3.0]), 2.0_f32);
    assert_eq!(center_of(vec![1.0_f64, 2.0, 3.0]), 2.0_f64);
}

/// The large dense fit learns finite parameters with the constant column at
/// unit scale.
#[test]
fn large_dense_fit_learns_finite_statistics() {
    let features = large_features();
    let model = default_scaler().fit(&features).expect("fit succeeds");
    assert_eq!(model.center().len(), 8);
    assert!(model.center().iter().all(|value| value.is_finite()));
    assert!(model.scale().iter().all(|value| value.is_finite()));
    assert!(model.scale().iter().take(7).all(|&value| value > 0.0));
    assert_eq!(model.scale()[7], 1.0);
}

/// The accessor helper is exercised on an exact vector.
#[test]
fn close_helper_matches_exact_vectors() {
    let values: Array1<f64> = array![1.0, 2.0, 3.0];
    assert_close(&values, &[1.0, 2.0, 3.0]);
}
