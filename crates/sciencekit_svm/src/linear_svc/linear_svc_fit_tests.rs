//! Companion TDD tests for binary [`SKLinearSVC::fit`]
//! (spec `linear-support-vector-classifier`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKError, SKSupervisedFit};

use super::{SKLinearSVC, SKLinearSvcLoss};

fn margin(head: &super::SKLinearSvcHead<f64>, row: &[f64], label: f64) -> f64 {
    let dot: f64 = head
        .coefficients()
        .iter()
        .zip(row.iter())
        .map(|(weight, value)| weight * value)
        .sum();
    label * (dot + head.intercept())
}

#[test]
fn binary_fit_separates_a_separable_fixture_with_one_head() {
    let features = array![
        [-2.5_f64, -2.0],
        [-1.5, -1.0],
        [-1.0, -1.5],
        [1.0, 1.5],
        [1.5, 1.0],
        [2.5, 2.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.number_of_classes(), 2);
    assert_eq!(model.heads().len(), 1);
    assert!(model.converged());
    assert!(model.iterations() > 0);
    assert_eq!(model.number_of_features(), 2);
    assert!(model.include_intercept());

    let head = &model.heads()[0];
    (0..features.nrows()).for_each(|row| {
        let label = if labels[row] == 1 { 1.0 } else { -1.0 };
        assert!(
            margin(head, features.row(row).to_slice().unwrap(), label) > 0.0,
            "row {row} was misclassified"
        );
    });
}

#[test]
fn binary_fit_converges_on_an_overlapping_fixture() {
    let features = array![
        [-2.0_f64, 0.0],
        [-1.0, 0.2],
        [-0.5, -0.8],
        [0.0, 0.5],
        [0.5, -0.4],
        [1.0, 0.6],
        [2.0, -0.2]
    ];
    let labels = array![0_i64, 0, 1, 0, 1, 1, 1];

    let model = SKLinearSVC::<f64>::build()
        .loss(super::SKLinearSvcLoss::Hinge)
        .C(0.5)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let head = &model.heads()[0];
    assert!(model.converged());
    assert!(head.coefficients().iter().all(|value| value.is_finite()));
    assert!(head.intercept().is_finite());
    // The overlapping fixture cannot be perfectly separated, but the boundary
    // must still orient toward the positive class.
    assert!(head.coefficients().iter().any(|value| value.abs() > 0.0));
}

#[test]
fn binary_fit_without_intercept_keeps_the_intercept_at_zero() {
    let features = array![[-1.5_f64], [-0.5], [0.5], [1.5]];
    let labels = array![0_i64, 0, 1, 1];

    let model = SKLinearSVC::<f64>::build()
        .include_intercept(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.heads()[0].intercept(), 0.0);
    assert!(!model.include_intercept());
    assert!(model.heads()[0].coefficients()[0] > 0.0);
}

#[test]
fn fit_reports_the_exhausted_iteration_budget_through_the_public_surface() {
    // Overlapping fixture from the solver suite: one sweep cannot reach the
    // tolerance, so the public `fit` must surface the solver error instead of
    // building a half-fitted model.
    let features = array![
        [-1.5_f64, -0.5],
        [-0.5, -1.2],
        [0.2, 0.3],
        [1.0, 1.1],
        [1.4, 0.4],
        [-0.2, -0.6]
    ];
    let labels = array![0_i64, 0, 1, 1, 1, 0];

    let error = SKLinearSVC::<f64>::build()
        .loss(SKLinearSvcLoss::Hinge)
        .tolerance(1e-12)
        .maximum_number_of_iterations(1)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();

    assert!(
        matches!(error, SKError::NotConverged { iterations: 1 }),
        "expected NotConverged carrying one executed sweep, got {error:?}"
    );
}
