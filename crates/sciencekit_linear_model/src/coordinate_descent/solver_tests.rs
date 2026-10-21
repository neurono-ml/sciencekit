//! Coordinate-descent solver tests: closed forms, KKT optimality, the
//! constraint/selection knobs and the dense/sparse agreement (spec
//! `coordinate-descent-solver`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::SKError;

use super::access::{SKDenseColumns, SKSparseColumns};
use super::config::{SKCoordinateDescentConfig, SKSelection};
use super::solution::SKCoordinateDescentSolution;
use super::solver::sk_solve_all_alphas;
use super::test_datasets::{
    TIGHT_TOLERANCE, assert_close, assert_close_scalar, one_feature, orthogonal_design, to_csr,
    zero_mean_feature,
};

/// A tight, screening-free configuration for the closed-form comparisons.
fn config() -> SKCoordinateDescentConfig<f64> {
    SKCoordinateDescentConfig {
        tolerance: TIGHT_TOLERANCE,
        maximum_number_of_iterations: 100_000,
        positive: false,
        selection: SKSelection::Cyclic,
        screening: false,
    }
}

/// Solve one dense design with a single penalty.
fn solve(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    alpha: f64,
    l1_ratio: f64,
    include_intercept: bool,
    config: &SKCoordinateDescentConfig<f64>,
) -> SKCoordinateDescentSolution<f64> {
    let access = SKDenseColumns::new(features.view());
    sk_solve_all_alphas(
        &access,
        targets.view(),
        &[alpha],
        l1_ratio,
        include_intercept,
        config,
    )
    .unwrap()
}

/// On an orthogonal mean-zero design the Lasso solution is `S(βⱼ, α)`.
#[test]
fn orthogonal_design_matches_the_closed_form() {
    let (features, targets) = orthogonal_design();
    let solution = solve(&features, &targets, 0.5, 1.0, true, &config());
    assert_close(
        &solution.coefficients.row(0).to_vec(),
        &[1.5, -0.5, 0.0],
        1e-8,
    );
    assert_close_scalar(solution.intercepts[0], 0.0, 1e-10);
}

/// A zero penalty recovers the ordinary-least-squares solution.
#[test]
fn zero_penalty_recovers_least_squares() {
    let (features, targets) = orthogonal_design();
    let solution = solve(&features, &targets, 0.0, 1.0, true, &config());
    assert_close(
        &solution.coefficients.row(0).to_vec(),
        &[2.0, -1.0, 0.0],
        1e-8,
    );
    assert_close_scalar(solution.intercepts[0], 0.0, 1e-10);
}

/// A penalty exceeding every gradient magnitude zeroes every coefficient.
#[test]
fn large_penalty_zeroes_every_coefficient() {
    let (features, targets) = orthogonal_design();
    let solution = solve(&features, &targets, 5.0, 1.0, true, &config());
    assert_close(
        &solution.coefficients.row(0).to_vec(),
        &[0.0, 0.0, 0.0],
        1e-12,
    );
    assert_close_scalar(solution.intercepts[0], 0.0, 1e-12);
}

/// The penalty shrinks the slope but leaves the unpenalized intercept intact.
#[test]
fn intercept_is_not_penalized() {
    let (features, targets) = zero_mean_feature();
    let weak = solve(&features, &targets, 1e-9, 1.0, true, &config());
    let strong = solve(&features, &targets, 10.0, 1.0, true, &config());
    assert!(strong.coefficients[[0, 0]].abs() < weak.coefficients[[0, 0]].abs());
    assert_close_scalar(weak.intercepts[0], 5.0, 1e-6);
    assert_close_scalar(strong.intercepts[0], 5.0, 1e-6);
}

/// The positive constraint clamps a negative coordinate at zero.
#[test]
fn positive_constraint_clamps_negative_coordinates() {
    let (features, targets) = orthogonal_design();
    let mut constrained = config();
    constrained.positive = true;
    let solution = solve(&features, &targets, 0.5, 1.0, true, &constrained);
    assert_close(
        &solution.coefficients.row(0).to_vec(),
        &[1.5, 0.0, 0.0],
        1e-8,
    );
}

/// The random selection converges to the same solution as the cyclic one.
#[test]
fn random_selection_matches_cyclic() {
    let (features, targets) = orthogonal_design();
    let cyclic = solve(&features, &targets, 0.5, 1.0, true, &config());
    let mut random = config();
    random.selection = SKSelection::Random;
    let randomized = solve(&features, &targets, 0.5, 1.0, true, &random);
    assert_close(
        &cyclic.coefficients.row(0).to_vec(),
        &randomized.coefficients.row(0).to_vec(),
        1e-8,
    );
}

/// A sparse design agrees with the equivalent dense design.
#[test]
fn sparse_matches_dense() {
    let (features, targets) = one_feature();
    let sparse = to_csr(&features);
    let dense_solution = solve(&features, &targets, 1.0, 1.0, false, &config());
    let access = SKSparseColumns::new(sparse.view());
    let sparse_solution =
        sk_solve_all_alphas(&access, targets.view(), &[1.0], 1.0, false, &config()).unwrap();
    // Σxy = 28, n·α = 3, Σx² = 14 → S(28, 3)/14 = 25/14.
    assert_close(
        &dense_solution.coefficients.row(0).to_vec(),
        &[25.0 / 14.0],
        1e-8,
    );
    assert_close(
        &dense_solution.coefficients.row(0).to_vec(),
        &sparse_solution.coefficients.row(0).to_vec(),
        1e-8,
    );
    assert_close_scalar(dense_solution.intercepts[0], 0.0, 1e-12);
}

/// The solver works on `f32` as well as `f64`.
#[test]
fn fits_both_supported_floats() {
    let features = array![[1.0f32], [2.0], [3.0]];
    let targets = array![2.0f32, 4.0, 6.0];
    let access = SKDenseColumns::new(features.view());
    let config = SKCoordinateDescentConfig {
        tolerance: 1e-5,
        maximum_number_of_iterations: 100_000,
        positive: false,
        selection: SKSelection::Cyclic,
        screening: false,
    };
    let solution =
        sk_solve_all_alphas(&access, targets.view(), &[1.0f32], 1.0, false, &config).unwrap();
    assert!((solution.coefficients[[0, 0]] - 25.0 / 14.0).abs() < 1e-3);
}

/// A zero iteration budget fails to converge immediately.
#[test]
fn zero_iteration_budget_fails_to_converge() {
    let (features, targets) = orthogonal_design();
    let mut exhausted = config();
    exhausted.maximum_number_of_iterations = 0;
    let access = SKDenseColumns::new(features.view());
    let error =
        sk_solve_all_alphas(&access, targets.view(), &[0.5], 1.0, true, &exhausted).unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations: 0 }));
}

/// A multi-alpha solve yields one independent row per penalty.
#[test]
fn multi_alpha_rows_are_independent() {
    let (features, targets) = orthogonal_design();
    let access = SKDenseColumns::new(features.view());
    let solution =
        sk_solve_all_alphas(&access, targets.view(), &[0.25, 1.5], 1.0, true, &config()).unwrap();
    assert_eq!(solution.coefficients.nrows(), 2);
    assert_close(
        &solution.coefficients.row(0).to_vec(),
        &[1.75, -0.75, 0.0],
        1e-8,
    );
    assert_close(
        &solution.coefficients.row(1).to_vec(),
        &[0.5, 0.0, 0.0],
        1e-8,
    );
}

/// The parallel multi-alpha path agrees with solving each penalty alone.
#[test]
fn parallel_multi_alpha_matches_sequential() {
    let (features, targets) = orthogonal_design();
    let access = SKDenseColumns::new(features.view());
    let alphas = [0.25, 0.5, 1.0, 1.5];
    let parallel =
        sk_solve_all_alphas(&access, targets.view(), &alphas, 1.0, true, &config()).unwrap();
    for (row_index, &alpha) in alphas.iter().enumerate() {
        let single = solve(&features, &targets, alpha, 1.0, true, &config());
        assert_close(
            &parallel.coefficients.row(row_index).to_vec(),
            &single.coefficients.row(0).to_vec(),
            1e-10,
        );
    }
}
