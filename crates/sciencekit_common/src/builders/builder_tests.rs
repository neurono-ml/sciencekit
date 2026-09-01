//! Tests for the mandatory builder foundation (spec `base-builders`) and the
//! composed automatic execution decision (spec `execution-decision`).

use super::reference_estimator::{SKReferenceEstimator, SKReferenceEstimatorBuilder};
use super::{SKBuilder, SKBuilderState, sk_validate_hyperparameter};
use crate::SKError;
use crate::execution::{SKAccessPattern, SKExecutionContext, SKExecutionMode, SKExecutionPlan};

/// Build a simulated context (never reads the physical machine).
fn simulated(
    memory: u64,
    cores: usize,
    dataset: u64,
    pattern: SKAccessPattern,
    batch: Option<usize>,
) -> SKExecutionContext {
    SKExecutionContext {
        available_memory_bytes: memory,
        cpu_cores: cores,
        dataset_size_bytes: dataset,
        access_pattern: pattern,
        batch_size_hint: batch,
    }
}

// ---- SKBuilderState (base-builders) --------------------------------------

/// A fresh builder state defaults its execution intent to `Automatic`.
#[test]
fn builder_state_defaults_to_automatic() {
    let state = SKBuilderState::new();
    assert_eq!(state.execution_intent(), SKExecutionMode::Automatic);
}

/// An explicit mode set on the state overrides the automatic default.
#[test]
fn builder_state_override_is_preserved() {
    let mut state = SKBuilderState::new();
    state.execution_mode(SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(
        state.execution_intent(),
        SKExecutionMode::OutOfCoreStreaming
    );
}

/// The validation helper accepts a valid hyperparameter value.
#[test]
fn validation_helper_accepts_valid_hyperparameter() {
    assert!(sk_validate_hyperparameter("samples", true, "unused").is_ok());
}

/// The validation helper reports an invalid hyperparameter by name and reason.
#[test]
fn validation_helper_reports_invalid_hyperparameter() {
    match sk_validate_hyperparameter("samples", false, "must be positive") {
        Err(SKError::InvalidHyperparameter { name, reason }) => {
            assert_eq!(name, "samples");
            assert_eq!(reason, "must be positive");
        }
        other => panic!("expected invalid hyperparameter, got {other:?}"),
    }
}

// ---- Automatic execution decision (execution-decision) --------------------

/// Automatic intent resolves to an in-memory plan for small data.
#[test]
fn automatic_intent_resolves_to_in_memory_plan() {
    let state = SKBuilderState::new();
    let context = simulated(1 << 30, 8, 100, SKAccessPattern::Sequential, None);
    let plan = state.resolve_plan(&context).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::InProcessSynchronous);
    assert_eq!(plan.parallelism, 8);
}

/// Automatic intent streams oversized sequential data with the declared batch.
#[test]
fn automatic_intent_streams_oversized_sequential_data() {
    let state = SKBuilderState::new();
    let context = simulated(1 << 30, 4, 1 << 40, SKAccessPattern::Sequential, Some(1024));
    let plan = state.resolve_plan(&context).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.batch_size, Some(1024));
}

/// An explicit incompatible intent fails with the precise error.
#[test]
fn explicit_incompatible_intent_is_rejected() {
    let mut state = SKBuilderState::new();
    state.execution_mode(SKExecutionMode::OutOfCoreStreaming);
    let context = simulated(1 << 30, 4, 1 << 40, SKAccessPattern::RandomAccess, None);
    match state.resolve_plan(&context) {
        Err(SKError::ExecutionModeIncompatible { mode, pattern }) => {
            assert_eq!(mode, "out-of-core-streaming");
            assert_eq!(pattern, "random-access");
        }
        other => panic!("expected incompatibility error, got {other:?}"),
    }
}

/// Concurrent resolutions are independent: identical inputs, identical plans.
#[test]
fn concurrent_resolutions_are_independent() {
    let context = simulated(1 << 30, 8, 1 << 40, SKAccessPattern::Sequential, Some(512));
    let expected = SKBuilderState::new().resolve_plan(&context).unwrap();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..16 {
            handles.push(scope.spawn(|| SKBuilderState::new().resolve_plan(&context).unwrap()));
        }
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}

// ---- Reference estimator (base-builders) ----------------------------------

/// The reference estimator is built exclusively through its builder.
#[test]
fn reference_estimator_is_built_through_its_builder() {
    let estimator = SKReferenceEstimatorBuilder::new()
        .minimum_samples(2)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap();
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
    assert_eq!(estimator.minimum_samples(), 2);
}

/// A default build carries the automatic execution intent.
#[test]
fn reference_estimator_defaults_to_automatic_execution() {
    let estimator = SKReferenceEstimatorBuilder::new().build().unwrap();
    assert_eq!(estimator.execution_intent(), SKExecutionMode::Automatic);
}

/// An explicit mode set on the builder overrides the default.
#[test]
fn reference_estimator_honors_explicit_mode_override() {
    let estimator = SKReferenceEstimatorBuilder::new()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(
        estimator.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}

/// An invalid hyperparameter surfaces as an error at build time.
#[test]
fn reference_estimator_builder_reports_invalid_hyperparameter() {
    match SKReferenceEstimatorBuilder::new()
        .minimum_samples(0)
        .build()
    {
        Err(SKError::InvalidHyperparameter { name, .. }) => {
            assert_eq!(name, "minimum_samples");
        }
        other => panic!("expected invalid hyperparameter, got {other:?}"),
    }
}

/// The trait setter remains usable through generic dispatch (`&mut self`).
#[test]
fn builder_trait_setter_is_usable_through_generic_dispatch() {
    fn configure<B: SKBuilder<SKReferenceEstimator>>(
        builder: &mut B,
        mode: SKExecutionMode,
    ) -> &mut B {
        builder.execution_mode(mode)
    }
    let mut builder = SKReferenceEstimatorBuilder::new();
    configure(&mut builder, SKExecutionMode::InProcessAsynchronous);
    let estimator = builder.build().unwrap();
    assert_eq!(
        estimator.execution_intent(),
        SKExecutionMode::InProcessAsynchronous
    );
}

/// The estimator resolves its own plan from its carried intent and a context.
#[test]
fn reference_estimator_resolves_its_plan_from_its_intent() {
    let estimator = SKReferenceEstimatorBuilder::new()
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap();
    let context = simulated(1 << 30, 4, 1 << 40, SKAccessPattern::Sequential, Some(256));
    let plan: SKExecutionPlan = estimator.resolve_plan(&context).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.batch_size, Some(256));
}
