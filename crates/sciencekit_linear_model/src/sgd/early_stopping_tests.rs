//! Companion tests for the early-stopping bookkeeping (spec
//! `stochastic-gradient-solver`, decision 7).

use ndarray::array;

use super::early_stopping::{SKEarlyStopping, sk_split_validation_indices};

#[test]
fn the_validation_slice_is_a_deterministic_prefix() {
    let (training, validation) = sk_split_validation_indices(10, 0.2_f64);
    assert_eq!(validation, vec![0, 1]);
    assert_eq!(training, vec![2, 3, 4, 5, 6, 7, 8, 9]);
    let mut all: Vec<usize> = training.into_iter().chain(validation).collect();
    all.sort_unstable();
    assert_eq!(all, (0..10).collect::<Vec<usize>>());
}

#[test]
fn a_tiny_design_keeps_at_least_one_validation_row() {
    let (_, validation) = sk_split_validation_indices(2, 0.9_f64);
    assert_eq!(validation.len(), 1);
    let (_, validation) = sk_split_validation_indices(1, 0.1_f64);
    assert!(validation.is_empty());
}

#[test]
fn the_best_iterate_is_kept() {
    let mut stopping = SKEarlyStopping::<f64>::new(0.1, 2);
    assert!(!stopping.record(array![1.0].view(), 0.0, 3.0));
    assert!(!stopping.record(array![2.0].view(), 1.0, 2.0));
    // Worsening by less than the tolerance counts as non-improving.
    assert!(!stopping.record(array![3.0].view(), 2.0, 2.05));
    assert_eq!(stopping.best_weights().unwrap()[0], 2.0);
    assert_eq!(stopping.best_intercept(), Some(1.0));
    assert_eq!(stopping.best_objective(), Some(2.0));
}

#[test]
fn stopping_needs_the_full_patience_of_non_improving_epochs() {
    let mut stopping = SKEarlyStopping::<f64>::new(0.01, 2);
    assert!(!stopping.record(array![1.0].view(), 0.0, 1.0));
    assert!(!stopping.record(array![1.1].view(), 0.0, 1.0));
    assert_eq!(stopping.epochs_without_improvement(), 1);
    assert!(stopping.record(array![1.2].view(), 0.0, 1.0));
    assert_eq!(stopping.epochs_without_improvement(), 2);
}

#[test]
fn an_improvement_resets_the_patience_counter() {
    let mut stopping = SKEarlyStopping::<f64>::new(0.01, 2);
    stopping.record(array![1.0].view(), 0.0, 1.0);
    stopping.record(array![1.1].view(), 0.0, 1.0);
    assert_eq!(stopping.epochs_without_improvement(), 1);
    assert!(!stopping.record(array![1.2].view(), 0.0, 0.5));
    assert_eq!(stopping.epochs_without_improvement(), 0);
    assert_eq!(stopping.best_objective(), Some(0.5));
}
