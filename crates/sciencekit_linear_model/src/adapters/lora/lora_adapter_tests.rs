//! Companion tests for the adapter configuration surface (task 1.1).

use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode};

use super::{SKLoraAdapterBuilder, SKLoraTargetModule};

#[test]
fn scaling_is_alpha_divided_by_rank() {
    let adapter = SKLoraAdapterBuilder::<f64>::new()
        .rank(8)
        .alpha(16.0)
        .build()
        .expect("rank 8, alpha 16 is valid");

    assert_eq!(adapter.rank(), 8);
    assert_eq!(adapter.alpha(), 16.0);
    assert!((adapter.scaling() - 2.0).abs() < 1e-12);
}

#[test]
fn scaling_holds_for_single_precision_too() {
    let adapter = SKLoraAdapterBuilder::<f32>::new()
        .rank(4)
        .alpha(3.0)
        .build()
        .expect("rank 4, alpha 3 is valid");

    assert!((adapter.scaling() - 0.75_f32).abs() < 1e-6);
}

#[test]
fn defaults_are_documented_and_stable() {
    let adapter = SKLoraAdapterBuilder::<f64>::new()
        .build()
        .expect("the defaults are valid");

    assert_eq!(adapter.identifier(), "default");
    assert_eq!(adapter.rank(), 4);
    assert_eq!(adapter.alpha(), 1.0);
    assert_eq!(adapter.target(), SKLoraTargetModule::LinearCoefficients);
    assert_eq!(adapter.backend(), SKBackendKind::Faer);
    assert_eq!(adapter.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn a_zero_rank_is_rejected_as_invalid_hyperparameter() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .rank(0)
        .build()
        .expect_err("rank zero has no low-rank factorization");

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "rank"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn a_non_finite_alpha_is_rejected_as_invalid_hyperparameter() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .alpha(f64::NAN)
        .build()
        .expect_err("NaN cannot scale an update");

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "alpha"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn an_empty_identifier_is_rejected_as_invalid_hyperparameter() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .identifier("")
        .build()
        .expect_err("an adapter must be addressable");

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "identifier"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn a_zero_iteration_budget_is_rejected_as_invalid_hyperparameter() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .maximum_number_of_iterations(0)
        .build()
        .expect_err("a fit needs at least one iteration");

    match error {
        SKError::InvalidHyperparameter { name, .. } => {
            assert_eq!(name, "maximum_number_of_iterations")
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn a_non_positive_learning_rate_is_rejected_as_invalid_hyperparameter() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .learning_rate(0.0)
        .build()
        .expect_err("a descent step needs a positive rate");

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "learning_rate"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn target_modules_round_trip_through_their_stable_names() {
    for target in SKLoraTargetModule::ALL {
        assert_eq!(
            SKLoraTargetModule::parse(target.parameter_name()),
            Some(target)
        );
        assert_eq!(
            SKLoraTargetModule::parse(target.tensor_segment()),
            Some(target)
        );
    }
    assert_eq!(SKLoraTargetModule::parse("unknown"), None);
}
