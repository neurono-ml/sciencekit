//! Tiny-data acceptance: the smallest usable problems and both losses
//! (PRD §8.7, spec `linear-support-vector-classifier`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use crate::SKLinearSVC;
use crate::linear_svc::SKLinearSvcLoss;

#[test]
fn acceptance_two_rows_fit_and_predict_for_both_losses() {
    let features = array![[-1.0_f64], [1.0]];
    let labels = array![0_i64, 1];

    [SKLinearSvcLoss::Hinge, SKLinearSvcLoss::SquaredHinge]
        .into_iter()
        .for_each(|loss| {
            let model = SKLinearSVC::<f64>::build()
                .loss(loss)
                .build()
                .unwrap()
                .fit(features.view(), labels.view())
                .unwrap();
            assert_eq!(model.loss(), loss);
            assert_eq!(model.heads().len(), 1);
            assert_eq!(model.predict_labels(features.view()).unwrap(), labels);
        });
}

#[test]
fn acceptance_tiny_multiclass_problem_retains_one_head_per_class() {
    let features = array![[-1.5_f64, 0.0], [1.5, 0.0], [0.0, 1.5]];
    let labels = array![0_i64, 1, 2];

    let model = SKLinearSVC::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.number_of_classes(), 3);
    assert_eq!(model.heads().len(), 3);
    assert_eq!(model.predict_labels(features.view()).unwrap(), labels);
}

#[test]
fn acceptance_single_feature_and_single_positive_class_are_supported() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let labels = array![0_i64, 0, 1, 1];

    let model = SKLinearSVC::<f64>::build()
        .C(2.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.predict_labels(features.view()).unwrap(), labels);
}

#[test]
fn acceptance_class_weights_run_on_a_tiny_imbalanced_problem() {
    use crate::svc::SKClassWeight;

    let features = array![[-2.0_f64], [-1.0], [-0.5], [0.2], [0.4], [0.8]];
    let labels = array![0_i64, 0, 0, 0, 1, 1];

    let model = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Balanced)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let predicted = model.predict_labels(features.view()).unwrap();
    // Balancing upweights the two positive rows, so the boundary may absorb the
    // closest negative row; every positive row must still be recovered.
    (0..labels.len())
        .filter(|&row| labels[row] == 1)
        .for_each(|row| {
            assert_eq!(predicted[row], 1);
        });
    let correct = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(left, right)| left == right)
        .count();
    assert!(correct >= 5);
    assert!(model.heads()[0].intercept().is_finite());
}
