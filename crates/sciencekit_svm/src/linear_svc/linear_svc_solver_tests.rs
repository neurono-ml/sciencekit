//! Companion TDD tests for the linear SVC dual coordinate-descent solver
//! (spec `linear-support-vector-classifier`).

use ndarray::array;

use super::config::SKLinearSvcLoss;
use super::design::SKDenseLinearSvcDesign;
use super::solver::sk_solve_linear_svc_dual;

fn solve_hinge(
    features: &ndarray::Array2<f64>,
    labels: &[f64],
    intercept: bool,
    tolerance: f64,
) -> (super::solver::SKLinearSvcSolverOutcome<f64>, usize) {
    let design = SKDenseLinearSvcDesign::new(features.view(), intercept);
    let weights = vec![1.0_f64; labels.len()];
    let outcome = sk_solve_linear_svc_dual(
        &design,
        labels,
        &weights,
        SKLinearSvcLoss::Hinge,
        1.0,
        tolerance,
        10_000,
    )
    .unwrap();
    let iterations = outcome.iterations;
    (outcome, iterations)
}

#[test]
fn hinge_dual_update_reaches_the_hand_computed_two_point_optimum() {
    // x = [-1], [+1], y = [-1], +1, no intercept. The dual optimum is the
    // boundary point alpha = (1, 0) because Q = [[1, -1], [-1, 1]] is singular;
    // it yields w = alpha_0 · y_0 · x_0 = 1.
    let features = array![[-1.0_f64], [1.0]];
    let labels = [-1.0_f64, 1.0];
    let (outcome, _) = solve_hinge(&features, &labels, false, 1e-12);

    assert!((outcome.coefficients[0] - 1.0).abs() < 1e-9);
    assert_eq!(outcome.iterations, 2);
}

#[test]
fn projected_gradient_stopping_rule_makes_tolerance_monotone() {
    let features = array![
        [-1.5_f64, -0.5],
        [-0.5, -1.2],
        [0.2, 0.3],
        [1.0, 1.1],
        [1.4, 0.4],
        [-0.2, -0.6]
    ];
    let labels = [-1.0_f64, -1.0, 1.0, 1.0, 1.0, -1.0];
    let (_, loose_iterations) = solve_hinge(&features, &labels, false, 0.5);
    let (tight, tight_iterations) = solve_hinge(&features, &labels, false, 1e-12);

    assert!(
        tight_iterations > loose_iterations,
        "tight tolerance ({tight_iterations}) must need more sweeps than the loose one ({loose_iterations})"
    );
    assert!(tight.coefficients[0] > 0.0);
    assert!(loose_iterations >= 1);
}

#[test]
fn intercept_is_recovered_for_the_two_point_problem() {
    // Augmented rows [-1, 1] and [1, 1] are orthogonal, so the dual optimum is
    // alpha = (1/2, 1/2): w = alpha_0·y_0·x_0 + alpha_1·y_1·x_1 = 1 and b = 0.
    let features = array![[-1.0_f64], [1.0]];
    let labels = [-1.0_f64, 1.0];
    let (outcome, _) = solve_hinge(&features, &labels, true, 1e-12);

    assert!((outcome.coefficients[0] - 1.0).abs() < 1e-5);
    assert!(outcome.coefficients[1].abs() < 1e-5);
    assert!(outcome.coefficients[1].abs() < 1e-5);
}
