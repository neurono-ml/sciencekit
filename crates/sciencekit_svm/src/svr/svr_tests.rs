//! Companion TDD tests for `SKSVR::fit` and `SKSVRModel::predict`
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_math::{SKKernel, SKKernelGamma};

use super::SKSVR;

#[test]
fn svr_fits_smooth_one_dimensional_response_and_retains_only_support_vectors() {
    let n = 20usize;
    let features =
        Array2::from_shape_fn((n, 1), |(r, _)| -2.0 + 4.0 * (r as f64) / ((n - 1) as f64));
    let targets = Array1::from_shape_fn(n, |r| features[[r, 0]].sin());

    let model = SKSVR::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Value(1.0))
        .C(10.0)
        .epsilon(0.05)
        .tolerance(1e-5)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();

    assert!(model.converged());
    assert_eq!(
        model.support_vectors().rows(),
        model.dual_coefficients().len()
    );
    assert!(model.support_vectors().rows() < n);

    let predictions = model.predict(features.view()).unwrap();
    assert_eq!(predictions.len(), n);
    predictions
        .iter()
        .zip(targets.iter())
        .for_each(|(&pred, &target)| assert!((pred - target).abs() <= 0.08));
}

#[test]
fn svr_widening_epsilon_band_never_increases_support_vector_count() {
    let residuals = [-0.45_f64, 0.02, -0.20, 0.01, 0.22, -0.02, 0.48, 0.0, -0.10, 0.09];
    let features = Array2::from_shape_fn((residuals.len(), 1), |(r, _)| -2.0 + r as f64 * 0.45);
    let targets = Array1::from_shape_fn(residuals.len(), |r| {
        1.8 * features[[r, 0]] - 0.3 + residuals[r]
    });

    let counts: Vec<usize> = [0.05_f64, 0.25, 0.60, 4.5]
        .into_iter()
        .map(|eps| {
            SKSVR::<f64>::build()
                .kernel(SKKernel::Linear)
                .C(25.0)
                .epsilon(eps)
                .tolerance(1e-6)
                .build()
                .unwrap()
                .fit(features.view(), targets.view())
                .unwrap()
                .support_vectors()
                .rows()
        })
        .collect();

    assert!(counts[1] <= counts[0], "counts = {counts:?}");
    assert!(counts[2] <= counts[1], "counts = {counts:?}");
    assert!(counts[3] <= counts[2], "counts = {counts:?}");
    assert_eq!(counts[3], 0);
}
