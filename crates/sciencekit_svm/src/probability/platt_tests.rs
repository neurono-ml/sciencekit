//! Companion TDD tests for Platt scaling (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::SKDataView;
use sciencekit_math::{SKKernel, SKKernelParameters};

use super::{SKPlattParameters, sk_calibrate_binary_platt, sk_fit_platt_sigmoid};
use crate::smo::SKSmoOptions;

#[test]
fn platt_sigmoid_matches_hand_computed_formula_bounds_and_monotonicity() {
    // Hand-computed check on explicit parameters A = -2.0, B = 0.5:
    // At f = 0.25 => A*f + B = 0.0 => P = 0.5.
    let explicit = SKPlattParameters::new(-2.0_f64, 0.5);
    assert!((explicit.predict_positive_probability(0.25) - 0.5).abs() < 1e-12);
    assert!(explicit.predict_positive_probability(1000.0) <= 1.0);
    assert!(explicit.predict_positive_probability(-1000.0) >= 0.0);

    // Fit Platt parameters on ordered binary margins and verify strict monotonicity.
    let margins = [-3.0_f64, -2.0, -1.2, -0.4, 0.5, 1.3, 2.2, 3.1];
    let signs = [-1_i8, -1, -1, -1, 1, 1, 1, 1];
    let fitted = sk_fit_platt_sigmoid(&margins, &signs);
    assert!(fitted.slope < 0.0, "expected negative slope, got {}", fitted.slope);

    let mut previous = 0.0_f64;
    for probe in [-5.0_f64, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0] {
        let prob = fitted.predict_positive_probability(probe);
        assert!((0.0..=1.0).contains(&prob));
        assert!(prob > previous);
        previous = prob;
    }
    assert!(fitted.predict_positive_probability(-3.0) < 0.3);
    assert!(fitted.predict_positive_probability(3.0) > 0.7);
}

#[test]
fn platt_five_fold_cross_validation_is_deterministic_and_monotonic() {
    let features = array![
        [-3.0_f64, -1.0],
        [-2.5, -0.8],
        [-2.0, -1.2],
        [-1.8, -0.5],
        [-1.5, -0.9],
        [1.5, 0.9],
        [1.8, 0.5],
        [2.0, 1.2],
        [2.5, 0.8],
        [3.0, 1.0]
    ];
    let signs = [-1_i8, -1, -1, -1, -1, 1, 1, 1, 1, 1];
    let bounds = [1.0_f64; 10];
    let parameters = SKKernelParameters::new(SKKernel::Linear, 0.0, 1, 0.0);
    let options = SKSmoOptions {
        tolerance: 1e-5,
        maximum_number_of_iterations: 500,
        cache_size_mib: 1,
        shrinking: true,
    };

    let first = sk_calibrate_binary_platt(
        SKDataView::Dense(features.view()),
        &signs,
        &bounds,
        parameters,
        options,
        77,
        0,
    )
    .unwrap();
    let second = sk_calibrate_binary_platt(
        SKDataView::Dense(features.view()),
        &signs,
        &bounds,
        parameters,
        options,
        77,
        0,
    )
    .unwrap();

    assert_eq!(first, second);
    assert!(first.slope < 0.0);
    assert!(first.predict_positive_probability(2.0) > first.predict_positive_probability(-2.0));
}
