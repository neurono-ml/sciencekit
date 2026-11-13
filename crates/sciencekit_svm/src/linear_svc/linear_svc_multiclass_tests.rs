//! Companion TDD tests for one-vs-rest multiclass fitting
//! (spec `linear-support-vector-classifier`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::SKLinearSVC;

fn multiclass_features() -> ndarray::Array2<f64> {
    array![
        [-3.0_f64, 0.0],
        [-2.5, 0.5],
        [-2.0, 0.0],
        [0.0, 3.0],
        [0.5, 2.5],
        [0.0, 2.0],
        [3.0, 0.0],
        [2.5, -0.5],
        [2.0, 0.0]
    ]
}

fn decision(head: &super::SKLinearSVCHead<f64>, row: &[f64]) -> f64 {
    let dot: f64 = head
        .coefficients()
        .iter()
        .zip(row.iter())
        .map(|(weight, value)| weight * value)
        .sum();
    dot + head.intercept()
}

#[test]
fn three_class_fit_retains_three_heads_in_canonical_order() {
    let features = multiclass_features();
    let labels = array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.number_of_classes(), 3);
    assert_eq!(model.heads().len(), 3);

    // Head `c` must score its own class highest on that class's rows.
    (0..features.nrows()).for_each(|row| {
        let own = labels[row] as usize;
        let own_decision = decision(&model.heads()[own], features.row(row).to_slice().unwrap());
        (0..3).filter(|&other| other != own).for_each(|other| {
            let other_decision =
                decision(&model.heads()[other], features.row(row).to_slice().unwrap());
            assert!(
                own_decision > other_decision,
                "row {row} of class {own} scored lower on its own head"
            );
        });
    });
}

#[test]
fn repeated_multiclass_fits_are_identical_and_keep_head_order() {
    let features = multiclass_features();
    let labels = array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];

    let fit_once = || {
        SKLinearSVC::<f64>::build()
            .C(1.5)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap()
    };
    let first = fit_once();
    let second = fit_once();

    first
        .heads()
        .iter()
        .zip(second.heads().iter())
        .for_each(|(left, right)| {
            assert_eq!(left.coefficients(), right.coefficients());
            assert_eq!(left.intercept(), right.intercept());
        });
}

#[test]
fn four_classes_retain_four_independent_heads() {
    let features = array![
        [-3.0_f64, 0.0],
        [-2.0, 0.0],
        [0.0, 3.0],
        [0.0, 2.0],
        [3.0, 0.0],
        [2.0, 0.0],
        [0.0, -3.0],
        [0.0, -2.0]
    ];
    let labels = array![0_i64, 0, 1, 1, 2, 2, 3, 3];

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.heads().len(), 4);
    assert!(
        model
            .heads()
            .iter()
            .all(|head| head.intercept().is_finite())
    );
}
