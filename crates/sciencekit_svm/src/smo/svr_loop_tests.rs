//! Companion TDD tests for the epsilon-SVR SMO loop
//! (spec `support-vector-machines`).

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sciencekit_math::{SKKernel, SKKernelParameters, sk_kernel_matrix};

use super::{SKSmoOptions, sk_solve_epsilon_svr};

#[test]
fn svr_smo_fits_smooth_one_dimensional_response_and_widening_epsilon_reduces_support_vectors() {
    let n = 15usize;
    let features =
        Array2::from_shape_fn((n, 1), |(r, _)| -2.0 + 4.0 * (r as f64) / ((n - 1) as f64));
    let targets: Vec<f64> = (0..n).map(|r| features[[r, 0]].sin()).collect();
    let parameters = SKKernelParameters::new(SKKernel::RadialBasisFunction, 1.0, 1, 0.0);
    let options = SKSmoOptions {
        tolerance: 1e-6,
        maximum_number_of_iterations: 4_000,
        cache_size: 1,
        shrinking: true,
    };

    let fitted = sk_solve_epsilon_svr(
        &targets,
        10.0,
        0.05,
        SKDataView::Dense(features.view()),
        parameters,
        options,
    )
    .unwrap();

    let gram = sk_kernel_matrix(
        &parameters,
        SKDataView::Dense(features.view()),
        SKDataView::Dense(features.view()),
    )
    .unwrap();
    for row in 0..n {
        let predicted: f64 = fitted
            .support_indices
            .iter()
            .zip(fitted.dual_coefficients.iter())
            .map(|(&sv, &coef)| coef * gram[[row, sv]])
            .sum::<f64>()
            + fitted.bias;
        assert!(
            (predicted - targets[row]).abs() <= 0.08,
            "row {row}: predicted {predicted}, target {}",
            targets[row]
        );
    }

    // Check epsilon-band support-vector monotonicity on a linear-kernel response
    // with graded residuals so widening epsilon absorbs points into the band.
    let offsets = [-0.40_f64, 0.02, -0.18, 0.0, 0.19, -0.01, 0.42, 0.01];
    let linear_x = Array2::from_shape_fn((offsets.len(), 1), |(r, _)| -1.5 + r as f64 * 0.4);
    let linear_y: Vec<f64> = (0..offsets.len())
        .map(|r| 1.5 * linear_x[[r, 0]] + 0.5 + offsets[r])
        .collect();
    let linear_params = SKKernelParameters::new(SKKernel::Linear, 0.0, 1, 0.0);

    let sv_narrow = sk_solve_epsilon_svr(
        &linear_y,
        50.0,
        0.05,
        SKDataView::Dense(linear_x.view()),
        linear_params,
        options,
    )
    .unwrap();
    let sv_medium = sk_solve_epsilon_svr(
        &linear_y,
        50.0,
        0.25,
        SKDataView::Dense(linear_x.view()),
        linear_params,
        options,
    )
    .unwrap();
    let sv_wide = sk_solve_epsilon_svr(
        &linear_y,
        50.0,
        1.0,
        SKDataView::Dense(linear_x.view()),
        linear_params,
        options,
    )
    .unwrap();

    assert!(sv_medium.support_indices.len() <= sv_narrow.support_indices.len());
    assert!(sv_wide.support_indices.len() <= sv_medium.support_indices.len());
}
