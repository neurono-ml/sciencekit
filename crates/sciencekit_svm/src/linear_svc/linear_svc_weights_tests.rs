//! Companion TDD tests for class weighting in the linear SVC
//! (spec `linear-support-vector-classifier`).

use ndarray::{Array1, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::{SKLinearSVC, SKLinearSvcLoss};
use crate::svc::SKClassWeight;

fn imbalanced_fixture() -> (ndarray::Array2<f64>, Array1<i64>) {
    let features = array![
        [-3.0_f64],
        [-2.5],
        [-2.0],
        [-1.5],
        [-1.0],
        [-0.5],
        [-0.2],
        [0.1],
        [0.3],
        [0.6]
    ];
    let labels = array![0_i64, 0, 0, 0, 0, 0, 0, 1, 1, 1];
    (features, labels)
}

#[test]
fn balanced_weights_change_the_boundary_on_an_imbalanced_fixture() {
    let (features, labels) = imbalanced_fixture();

    let unweighted = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let balanced = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Balanced)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let unweighted_head = &unweighted.heads()[0];
    let balanced_head = &balanced.heads()[0];
    assert!(
        (unweighted_head.intercept() - balanced_head.intercept()).abs() > 1e-6
            || (unweighted_head.coefficients()[0] - balanced_head.coefficients()[0]).abs() > 1e-6,
        "balanced weights must change the fitted boundary"
    );

    // The minority class is the positive one; balancing may not lose a single
    // minority training row.
    let balanced_predictions = balanced.predict_labels(features.view()).unwrap();
    (0..labels.len())
        .filter(|&row| labels[row] == 1)
        .for_each(|row| {
            assert_eq!(balanced_predictions[row], 1, "row {row} lost recall");
        });
}

#[test]
fn explicit_weights_equal_to_one_reproduce_the_unweighted_model() {
    let (features, labels) = imbalanced_fixture();

    let unweighted = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let explicit_unit = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 1.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(
        unweighted.heads()[0].coefficients(),
        explicit_unit.heads()[0].coefficients()
    );
    assert_eq!(
        unweighted.heads()[0].intercept(),
        explicit_unit.heads()[0].intercept()
    );
}

#[test]
fn explicit_class_weights_are_honored_and_change_the_fit() {
    let (features, labels) = imbalanced_fixture();

    let unit = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 1.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let upweighted = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 4.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert!(
        (unit.heads()[0].intercept() - upweighted.heads()[0].intercept()).abs() > 1e-6
            || (unit.heads()[0].coefficients()[0] - upweighted.heads()[0].coefficients()[0]).abs()
                > 1e-6,
        "explicit class weights must reach the solver"
    );
}

#[test]
fn label_map_weights_resolve_against_the_canonical_class_table() {
    let (features, labels) = imbalanced_fixture();

    let label_map = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::from_labels([(0_i64, 1.0_f64), (1, 4.0)]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let explicit = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 4.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(
        label_map.heads()[0].coefficients(),
        explicit.heads()[0].coefficients()
    );
    assert_eq!(
        label_map.heads()[0].intercept(),
        explicit.heads()[0].intercept()
    );
}

#[test]
fn explicit_zero_weight_class_is_ignored_without_breaking_convergence() {
    // `Explicit` accepts a zero weight (the validator only requires finite,
    // non-negative values with at least one positive entry). A zero-weighted
    // class contributes nothing to the objective, so the solver must drop its
    // rows instead of reporting their permanently violating gradient.
    let features = array![[-2.0_f64], [-1.0], [-0.5], [0.2], [0.4], [0.8]];
    let labels = array![0_i64, 0, 0, 0, 1, 1];

    [SKLinearSvcLoss::Hinge, SKLinearSvcLoss::SquaredHinge]
        .into_iter()
        .for_each(|loss| {
            let model = SKLinearSVC::<f64>::build()
                .loss(loss)
                .class_weight(SKClassWeight::Explicit(vec![0.0, 1.0]))
                .build()
                .unwrap()
                .fit(features.view(), labels.view())
                .unwrap_or_else(|error| {
                    panic!("zero-weight class must not break the {loss:?} fit: {error:?}")
                });

            assert!(
                model.heads()[0]
                    .coefficients()
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!(model.heads()[0].intercept().is_finite());
        });
}
