//! Companion tests for the imputer builder (spec `simple-imputer`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::core_implementation::SKSimpleImputer;
use super::strategy::SKImputationStrategy;

#[test]
fn default_configuration_is_the_documented_one() {
    let imputer: SKSimpleImputer<f64> = SKSimpleImputer::build().build().unwrap();
    assert_eq!(imputer.strategy(), SKImputationStrategy::Mean);
    assert!(
        imputer.missing_values().is_nan(),
        "the default marker is NaN"
    );
    assert_eq!(imputer.fill_value(), None);
    assert!(!imputer.keep_empty_features());
    assert!(!imputer.add_indicator());
    assert_eq!(imputer.backend(), SKBackendKind::Faer);
    assert_eq!(imputer.execution_intent(), SKExecutionMode::Automatic);
}

#[test]
fn every_setter_is_chainable_and_reaches_the_estimator() {
    let imputer: SKSimpleImputer<f64> = SKSimpleImputer::build()
        .strategy(SKImputationStrategy::Constant)
        .missing_values(0.0)
        .fill_value(Some(1.5))
        .keep_empty_features(true)
        .add_indicator(true)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(imputer.strategy(), SKImputationStrategy::Constant);
    assert_eq!(imputer.missing_values(), 0.0);
    assert_eq!(imputer.fill_value(), Some(1.5));
    assert!(imputer.keep_empty_features());
    assert!(imputer.add_indicator());
    assert_eq!(imputer.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        imputer.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn generic_over_the_scalar_for_f32() {
    let imputer: SKSimpleImputer<f32> = SKSimpleImputer::build().build().unwrap();
    assert!(imputer.missing_values().is_nan());
    assert_eq!(imputer.strategy(), SKImputationStrategy::Mean);
}

#[test]
fn fill_value_of_nan_under_constant_is_rejected() {
    let result: Result<SKSimpleImputer<f64>, SKError> = SKSimpleImputer::build()
        .strategy(SKImputationStrategy::Constant)
        .fill_value(Some(f64::NAN))
        .build();
    let error = result.err().expect("filling with NaN must be rejected");
    assert!(
        error.to_string().contains("fill_value"),
        "the error names the offending parameter: {error}"
    );
}

#[test]
fn trait_execution_mode_setter_matches_the_chainable_one() {
    let mut builder = SKSimpleImputer::<f64>::build();
    SKBuilder::<SKSimpleImputer<f64>>::execution_mode(
        &mut builder,
        SKExecutionMode::OutOfCoreStreaming,
    );
    let imputer = builder.build().unwrap();
    assert_eq!(
        imputer.execution_intent(),
        SKExecutionMode::OutOfCoreStreaming
    );
}
