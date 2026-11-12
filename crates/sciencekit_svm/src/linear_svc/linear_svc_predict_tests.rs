//! Companion TDD tests for the decision function and label prediction
//! (spec `linear-support-vector-classifier`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::SKLinearSVC;

#[test]
fn decision_function_shape_and_labels_match_the_argmax() {
    let features = array![
        [-3.0_f64, 0.0],
        [-2.5, 0.5],
        [-2.0, 0.0],
        [0.0, 3.0],
        [0.5, 2.5],
        [0.0, 2.0],
        [3.0, 0.0],
        [2.5, -0.5],
        [2.0, 0.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let decisions = model.decision_function(features.view()).unwrap();
    assert_eq!(decisions.dim(), (9, 3));

    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted.len(), 9);
    assert_eq!(predicted, labels);

    decisions
        .rows()
        .into_iter()
        .zip(predicted.iter())
        .for_each(|(row, &label)| {
            let argmax = row
                .iter()
                .enumerate()
                .fold(
                    0usize,
                    |winner, (index, &value)| {
                        if value > row[winner] { index } else { winner }
                    },
                );
            assert_eq!(argmax as i64, label);
        });
}

#[test]
fn binary_decision_function_exposes_two_mirrored_columns() {
    let features = array![[-2.0_f64, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]];
    let labels = array![0_i64, 0, 1, 1];

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let decisions = model.decision_function(features.view()).unwrap();
    assert_eq!(decisions.dim(), (4, 2));
    decisions.rows().into_iter().for_each(|row| {
        assert!((row[0] + row[1]).abs() < 1e-12);
    });
    assert_eq!(model.predict_labels(features.view()).unwrap(), labels);
}

#[test]
fn decision_ties_break_by_the_smallest_canonical_class_index() {
    use ndarray::Array1;

    let three_features = array![[-2.0_f64, 0.0], [0.0, 2.0], [2.0, 0.0]];
    let three = array![0_i64, 1, 2];
    let mut model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(three_features.view(), three.view())
        .unwrap();
    model.heads[0].coefficients = Array1::zeros(2);
    model.heads[1].coefficients = Array1::zeros(2);
    model.heads[2].coefficients = Array1::zeros(2);
    model.heads[0].intercept = 1.0;
    model.heads[1].intercept = 1.0;
    model.heads[2].intercept = 0.0;

    let predicted = model.predict_labels(three_features.view()).unwrap();
    assert_eq!(predicted, array![0_i64, 0, 0]);
    let decisions = model.decision_function(three_features.view()).unwrap();
    assert!((decisions[[0, 0]] - decisions[[0, 1]]).abs() < 1e-12);
}
