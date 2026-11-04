//! Companion tests for the running epoch objective (spec
//! `stochastic-gradient-solver`, decision 7).

use super::objective::{SKEpochObjective, sk_objective_improved};

#[test]
fn an_empty_objective_is_zero() {
    let objective = SKEpochObjective::<f64>::new();
    assert_eq!(objective.mean(), 0.0);
    assert_eq!(objective.count(), 0);
}

#[test]
fn the_running_loss_equals_the_mean_of_the_batch_losses() {
    let mut objective = SKEpochObjective::<f64>::new();
    for loss in [1.0, 2.0, 3.0] {
        objective.record(loss);
    }
    assert_eq!(objective.count(), 3);
    assert!((objective.mean() - 2.0).abs() < 1e-12);
}

#[test]
fn improvement_is_measured_beyond_the_tolerance() {
    assert!(sk_objective_improved(10.0_f64, 9.0, 0.5));
    assert!(!sk_objective_improved(10.0_f64, 9.8, 0.5));
    assert!(!sk_objective_improved(10.0_f64, 10.0, 0.5));
    assert!(!sk_objective_improved(10.0_f64, 11.0, 0.5));
}
