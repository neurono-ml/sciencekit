//! Companion tests for the logistic configuration, resolution and class-weight
//! substrate (spec `logistic-solver`).

use super::class_weight::sk_expand_class_weights;
use super::config::{
    SKClassWeight, SKLogisticPenalty, SKLogisticSolver, SKMultiClass,
    sk_validate_logistic_hyperparameters,
};
use super::resolution::{
    SKResolvedMultiClass, sk_resolve_multi_class, sk_validate_solver_penalty,
};
use sciencekit_common::SKError;

#[test]
fn valid_hyperparameters_are_accepted() {
    assert!(
        sk_validate_logistic_hyperparameters(1.0_f64, 1e-4, 100, 0.5).is_ok(),
        "a well-formed configuration must validate"
    );
}

#[test]
fn non_positive_strength_is_rejected() {
    let error = sk_validate_logistic_hyperparameters(0.0_f64, 1e-4, 100, 0.5).unwrap_err();
    assert!(
        matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "C"),
        "a non-positive regularization strength must name C"
    );
}

#[test]
fn zero_tolerance_and_budget_are_rejected() {
    assert!(sk_validate_logistic_hyperparameters(1.0_f64, 0.0, 100, 0.5).is_err());
    assert!(sk_validate_logistic_hyperparameters(1.0_f64, 1e-4, 0, 0.5).is_err());
}

#[test]
fn out_of_range_l1_ratio_is_rejected() {
    let error = sk_validate_logistic_hyperparameters(1.0_f64, 1e-4, 100, 1.5).unwrap_err();
    assert!(
        matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "l1_ratio"),
        "an out-of-range l1_ratio must be named"
    );
}

#[test]
fn supported_solver_penalty_pairs_are_accepted() {
    let accepted = [
        (SKLogisticSolver::Lbfgs, SKLogisticPenalty::L2),
        (SKLogisticSolver::Lbfgs, SKLogisticPenalty::None),
        (SKLogisticSolver::Liblinear, SKLogisticPenalty::L1),
        (SKLogisticSolver::Liblinear, SKLogisticPenalty::L2),
        (SKLogisticSolver::Saga, SKLogisticPenalty::L1),
        (SKLogisticSolver::Saga, SKLogisticPenalty::L2),
        (SKLogisticSolver::Saga, SKLogisticPenalty::ElasticNet),
        (SKLogisticSolver::Saga, SKLogisticPenalty::None),
    ];
    for (solver, penalty) in accepted {
        assert!(
            sk_validate_solver_penalty(solver, penalty).is_ok(),
            "{solver:?} with {penalty:?} must be accepted"
        );
    }
}

#[test]
fn unsupported_solver_penalty_pairs_are_rejected() {
    let rejected = [
        (SKLogisticSolver::Lbfgs, SKLogisticPenalty::L1),
        (SKLogisticSolver::Lbfgs, SKLogisticPenalty::ElasticNet),
        (SKLogisticSolver::Liblinear, SKLogisticPenalty::ElasticNet),
        (SKLogisticSolver::Liblinear, SKLogisticPenalty::None),
    ];
    for (solver, penalty) in rejected {
        let error = sk_validate_solver_penalty(solver, penalty).unwrap_err();
        assert!(
            matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "penalty"),
            "{solver:?} with {penalty:?} must be rejected naming the penalty"
        );
    }
}

#[test]
fn a_single_class_is_rejected() {
    let error =
        sk_resolve_multi_class(SKMultiClass::Auto, SKLogisticSolver::Lbfgs, 1).unwrap_err();
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "targets"));
}

#[test]
fn two_classes_always_fit_one_binary_head() {
    for strategy in [
        SKMultiClass::Auto,
        SKMultiClass::OneVersusRest,
        SKMultiClass::Multinomial,
    ] {
        assert_eq!(
            sk_resolve_multi_class(strategy, SKLogisticSolver::Lbfgs, 2).unwrap(),
            SKResolvedMultiClass::Binary
        );
    }
}

#[test]
fn auto_resolves_by_solver() {
    assert_eq!(
        sk_resolve_multi_class(SKMultiClass::Auto, SKLogisticSolver::Lbfgs, 3).unwrap(),
        SKResolvedMultiClass::Multinomial
    );
    assert_eq!(
        sk_resolve_multi_class(SKMultiClass::Auto, SKLogisticSolver::Saga, 3).unwrap(),
        SKResolvedMultiClass::Multinomial
    );
    assert_eq!(
        sk_resolve_multi_class(SKMultiClass::Auto, SKLogisticSolver::Liblinear, 3).unwrap(),
        SKResolvedMultiClass::OneVersusRest
    );
}

#[test]
fn liblinear_with_multinomial_is_rejected() {
    let error = sk_resolve_multi_class(
        SKMultiClass::Multinomial,
        SKLogisticSolver::Liblinear,
        3,
    )
    .unwrap_err();
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "multi_class"));
}

#[test]
fn none_weighting_weighs_every_sample_one() {
    let weights = sk_expand_class_weights(&SKClassWeight::<f64>::None, &[0, 1, 1], 2).unwrap();
    assert_eq!(weights, vec![1.0, 1.0, 1.0]);
}

#[test]
fn balanced_weighting_balances_the_classes() {
    // Class 0 has one sample, class 1 has three: the balanced weights must make
    // each class's total weight equal.
    let weights =
        sk_expand_class_weights(&SKClassWeight::<f64>::Balanced, &[0, 1, 1, 1], 2).unwrap();
    let class_zero: f64 = weights[0];
    let class_one: f64 = weights[1..].iter().sum();
    assert!(
        (class_zero - class_one).abs() < 1e-12,
        "balanced weights must equalize the class totals, got {class_zero} and {class_one}"
    );
}

#[test]
fn explicit_weighting_is_applied_in_class_order() {
    let weights =
        sk_expand_class_weights(&SKClassWeight::Explicit(vec![2.0, 5.0]), &[1, 0, 1], 2).unwrap();
    assert_eq!(weights, vec![5.0, 2.0, 5.0]);
}

#[test]
fn negative_and_all_zero_explicit_weights_are_rejected() {
    let negative = sk_expand_class_weights(&SKClassWeight::Explicit(vec![1.0, -1.0]), &[0], 2);
    assert!(matches!(
        negative.unwrap_err(),
        SKError::InvalidHyperparameter { name, .. } if name == "class_weight"
    ));
    let all_zero = sk_expand_class_weights(&SKClassWeight::Explicit(vec![0.0, 0.0]), &[0], 2);
    assert!(all_zero.is_err());
}

#[test]
fn explicit_weight_length_must_match_the_class_count() {
    let error = sk_expand_class_weights(&SKClassWeight::Explicit(vec![1.0]), &[0], 3).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
