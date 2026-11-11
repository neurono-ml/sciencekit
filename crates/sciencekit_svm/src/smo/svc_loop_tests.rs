//! Companion TDD tests for the binary C-SVC SMO loop
//! (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::SKDataView;
use sciencekit_math::{SKKernel, SKKernelParameters};

use super::{SKSmoOptions, sk_solve_binary_c_svc};

#[test]
fn svc_smo_matches_hand_computed_two_point_dual_solution() {
    // Two points in 1-D: x_0 = [2.0] (y_0 = -1), x_1 = [4.0] (y_1 = +1), C = 10.
    // Optimal hyperplane: w = 1.0, b = -3.0.
    // Dual: alpha_0 = 0.5, alpha_1 = 0.5 => dual_coefficients = [-0.5, +0.5].
    let features = array![[2.0_f64], [4.0]];
    let signs = [-1_i8, 1];
    let bounds = [10.0_f64, 10.0];
    let parameters = SKKernelParameters::new(SKKernel::Linear, 0.0, 1, 0.0);
    let options = SKSmoOptions {
        tolerance: 1e-8,
        maximum_number_of_iterations: 100,
        cache_size_mib: 1,
        shrinking: true,
    };

    let result = sk_solve_binary_c_svc(
        &signs,
        &bounds,
        SKDataView::Dense(features.view()),
        parameters,
        options,
    )
    .unwrap();

    assert_eq!(result.support_indices, vec![0, 1]);
    assert!((result.dual_coefficients[0] - (-0.5)).abs() < 1e-8);
    assert!((result.dual_coefficients[1] - 0.5).abs() < 1e-8);
    assert!((result.bias - (-3.0)).abs() < 1e-8);
}

#[test]
fn svc_smo_separates_small_separable_dataset_with_clipping_and_shrinking() {
    let features = array![
        [-3.0_f64, -2.0],
        [-2.0, -1.5],
        [-1.0, -1.0],
        [1.0, 1.0],
        [2.0, 1.5],
        [3.0, 2.5]
    ];
    let signs = [-1_i8, -1, -1, 1, 1, 1];
    let bounds = [0.25_f64; 6];
    let parameters = SKKernelParameters::new(SKKernel::Linear, 0.0, 1, 0.0);

    for shrinking in [false, true] {
        let options = SKSmoOptions {
            tolerance: 1e-6,
            maximum_number_of_iterations: 500,
            cache_size_mib: 1,
            shrinking,
        };
        let result = sk_solve_binary_c_svc(
            &signs,
            &bounds,
            SKDataView::Dense(features.view()),
            parameters,
            options,
        )
        .unwrap();

        // Check box clipping |y_i * alpha_i| <= 0.25 and correct classification on all 6 rows.
        result
            .dual_coefficients
            .iter()
            .for_each(|&coef| assert!(coef.abs() <= 0.25 + 1e-10));

        for (row_index, &label) in signs.iter().enumerate() {
            let decision: f64 = result
                .support_indices
                .iter()
                .zip(result.dual_coefficients.iter())
                .map(|(&sv_idx, &coef)| {
                    let dot = features.row(sv_idx).dot(&features.row(row_index));
                    coef * dot
                })
                .sum::<f64>()
                + result.bias;
            assert!(decision * f64::from(label) > 0.0);
        }
    }
}
