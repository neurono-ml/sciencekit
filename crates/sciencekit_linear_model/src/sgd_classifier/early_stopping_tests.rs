//! Companion tests for early stopping in the SGD classifier (spec
//! `stochastic-gradient-solver`, decision 7).

use ndarray::{Array1, Array2};
use sciencekit_common::SKSupervisedFit;
use sciencekit_common::builders::SKBuilder;

use super::{SKSGDClassifier, SKSGDClassifierModel};
use crate::sgd::{SKConvergencePolicy, SKSGDClassifierLoss, SKSGDLearningRate};

/// A single-feature binary design whose validation prefix carries the opposite
/// label of the training remainder at the same positive feature values, so
/// every training epoch raises the validation hinge loss.
fn anti_correlated_design() -> (Array2<f64>, Array1<i64>) {
    let values = [
        0.5, 0.75, 1.0, 1.25, 1.5, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
        3.25, 3.5, 3.75, 4.0,
    ];
    let features = Array2::from_shape_vec((values.len(), 1), values.to_vec())
        .expect("the design is rectangular");
    let targets = Array1::from_iter(
        values
            .iter()
            .enumerate()
            .map(|(row, _)| if row < 5 { 0 } else { 1 }),
    );
    (features, targets)
}

fn fit(patience: usize, budget: usize) -> SKSGDClassifierModel<f64> {
    let (features, targets) = anti_correlated_design();
    SKSGDClassifier::<f64>::build()
        .loss(SKSGDClassifierLoss::Hinge)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.2)
        .alpha(0.0)
        .shuffle(false)
        .early_stopping(true)
        .validation_fraction(0.25)
        .number_of_epochs_without_improvement(patience)
        .maximum_number_of_iterations(budget)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
}

#[test]
fn early_stopping_keeps_the_best_validation_iterate() {
    let early = fit(1, 10);
    assert!(
        early.converged(),
        "the patience stop must report convergence"
    );
    assert!(early.epochs_taken() <= 10);
    assert_eq!(early.epochs_taken(), 2);

    let two_epochs = fit(10, 2);
    assert_eq!(two_epochs.epochs_taken(), 2);
    assert_eq!(early.coefficients(), two_epochs.coefficients());
    assert_ne!(
        two_epochs.coefficients().row(0),
        two_epochs.states[0].weights.view()
    );
    assert_eq!(early.coefficients().row(0), early.states[0].weights.view());
}
