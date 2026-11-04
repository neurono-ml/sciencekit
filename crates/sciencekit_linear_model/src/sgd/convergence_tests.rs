//! Companion tests for the convergence bookkeeping (spec
//! `stochastic-gradient-solver`, decision 7).

use sciencekit_common::SKError;

use super::convergence::{SKConvergencePolicy, SKConvergenceTracker, sk_finish_epochs};

#[test]
fn a_plateau_converges_before_the_budget() {
    let mut tracker = SKConvergenceTracker::<f64>::new(0.1, 2);
    assert!(!tracker.record(5.0));
    assert!(!tracker.record(4.0));
    // Two consecutive epochs improving by no more than the tolerance.
    assert!(!tracker.record(3.95));
    assert!(tracker.record(3.90));
    assert!(tracker.converged());
    assert_eq!(tracker.epochs_without_improvement(), 2);
}

#[test]
fn strict_exhaustion_raises_not_converged_with_the_iterations() {
    let error = sk_finish_epochs(SKConvergencePolicy::Strict, 10, 10, false, false).unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations } if iterations == 10));
}

#[test]
fn permissive_exhaustion_returns_the_last_iterate() {
    let converged =
        sk_finish_epochs(SKConvergencePolicy::Permissive, 10, 10, false, false).unwrap();
    assert!(!converged);
}

#[test]
fn a_provider_limited_fit_never_raises_even_under_strict() {
    let converged = sk_finish_epochs(SKConvergencePolicy::Strict, 1000, 1, false, true).unwrap();
    assert!(!converged);
}

#[test]
fn an_early_stopped_fit_returns_without_raising() {
    let converged = sk_finish_epochs(SKConvergencePolicy::Strict, 100, 4, false, false).unwrap();
    assert!(!converged);
}

#[test]
fn a_converged_fit_reports_success_even_under_strict() {
    assert!(sk_finish_epochs(SKConvergencePolicy::Strict, 10, 3, true, false).unwrap());
}

#[test]
fn strict_is_the_default_policy() {
    assert_eq!(SKConvergencePolicy::default(), SKConvergencePolicy::Strict);
}
