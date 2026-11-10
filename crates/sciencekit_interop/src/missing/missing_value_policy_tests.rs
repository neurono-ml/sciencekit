//! Tests for the documented missing-value policy (spec
//! `dataset-interop-core`).

use sciencekit_common::{SK_CATEGORICAL_MISSING_CODE, SKError};

use super::SKMissingValuePolicy;

/// A caller who does not choose a policy gets the safe default.
#[test]
fn default_policy_rejects() {
    let policy = SKMissingValuePolicy::default();
    assert_eq!(policy, SKMissingValuePolicy::Reject);
    assert!(policy.is_reject());
    assert_eq!(policy.name(), "reject");
}

/// Constant fill validates its value and keeps a finite one.
#[test]
fn constant_fill_validates_its_value() {
    let policy = SKMissingValuePolicy::constant_fill(0.0).unwrap();
    assert_eq!(policy.fill_value(), Some(0.0));
    assert_eq!(policy.name(), "constant_fill");
    assert!(!policy.is_reject());

    let negative = SKMissingValuePolicy::constant_fill(-3.5).unwrap();
    assert_eq!(negative.fill_value(), Some(-3.5));
}

/// A non-finite constant is rejected instead of mimicking a sentinel.
#[test]
fn constant_fill_rejects_non_finite_values() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = SKMissingValuePolicy::constant_fill(invalid).unwrap_err();
        match &error {
            SKError::InvalidHyperparameter { name, reason } => {
                assert_eq!(*name, "constant_fill_value");
                assert!(reason.contains("finite"));
            }
            other => panic!("expected an invalid-hyperparameter error, got {other:?}"),
        }
    }
}

/// The categorical policy points at the library's documented missing code.
#[test]
fn categorical_policy_uses_the_documented_code() {
    let policy = SKMissingValuePolicy::CategoricalMissingCode;
    assert_eq!(SK_CATEGORICAL_MISSING_CODE, -1);
    assert_eq!(policy.name(), "categorical_missing_code");
    assert_eq!(policy.fill_value(), None);
    assert!(!policy.is_reject());
}

/// Policy names are stable for spans and error messages.
#[test]
fn policy_names_are_stable() {
    assert_eq!(SKMissingValuePolicy::Reject.to_string(), "reject");
    assert_eq!(SKMissingValuePolicy::NumericNaN.to_string(), "numeric_nan");
    assert_eq!(
        SKMissingValuePolicy::CategoricalMissingCode.to_string(),
        "categorical_missing_code"
    );
    assert_eq!(
        SKMissingValuePolicy::ConstantFill(1.0).to_string(),
        "constant_fill"
    );
}
