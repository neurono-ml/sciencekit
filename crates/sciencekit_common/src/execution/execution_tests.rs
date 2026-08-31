//! Tests for execution planning (spec `execution-planning`).

use super::{
    SKAccessPattern, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    sk_resolve_execution_plan,
};
use crate::SKError;

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

/// Automatic intent over a known context produces a concrete plan.
#[test]
fn automatic_intent_produces_concrete_plan() {
    // Dataset (100 bytes) fits in memory (1 GiB).
    let ctx = simulated(1 << 30, 8, 100, SKAccessPattern::Sequential, None);
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::InProcessSynchronous);
    assert_eq!(plan.parallelism, 8);
}

/// Automatic intent over a larger-than-memory sequential dataset streams.
#[test]
fn automatic_oversized_sequential_streams() {
    let ctx = simulated(1 << 30, 4, 1 << 40, SKAccessPattern::Sequential, Some(1024));
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.batch_size, Some(1024));
}

/// Automatic intent over a larger-than-memory random-access dataset maps.
#[test]
fn automatic_oversized_random_maps() {
    let ctx = simulated(1 << 30, 4, 1 << 40, SKAccessPattern::RandomAccess, None);
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreMemoryMapped);
}

/// Explicit intent compatible with the pattern is preserved in the plan.
#[test]
fn explicit_compatible_intent_is_preserved() {
    let ctx = simulated(1 << 30, 2, 1 << 40, SKAccessPattern::Sequential, None);
    let plan = sk_resolve_execution_plan(SKExecutionMode::OutOfCoreStreaming, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
}

/// Same intent + context → identical plans (deterministic, pure).
#[test]
fn same_context_produces_same_plan() {
    let ctx = simulated(1 << 30, 6, 1 << 40, SKAccessPattern::Sequential, Some(64));
    let a = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    let b = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(a, b);
}

/// Simulated context drives resolution without reading the physical machine.
#[test]
fn simulated_context_dispenses_with_real_machine() {
    let ctx = simulated(1, 1, 10, SKAccessPattern::Sequential, None);
    // With only 1 byte of memory and a 10-byte dataset, automatic must stream.
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.parallelism, 1);
}

/// Explicit streaming on a random-access algorithm fails with a precise error.
#[test]
fn sequential_streaming_refused_for_random_access() {
    let ctx = simulated(1 << 30, 4, 1 << 40, SKAccessPattern::RandomAccess, None);
    let result = sk_resolve_execution_plan(SKExecutionMode::OutOfCoreStreaming, &ctx);
    match result {
        Err(SKError::ExecutionModeIncompatible { mode, pattern }) => {
            assert_eq!(mode, "out-of-core-streaming");
            assert_eq!(pattern, "random-access");
        }
        other => panic!("expected incompatibility error, got {other:?}"),
    }
}

/// Automatic intent never produces the incompatibility error.
#[test]
fn automatic_never_conflicts_with_declared_pattern() {
    for pattern in [
        SKAccessPattern::Sequential,
        SKAccessPattern::RandomAccess,
        SKAccessPattern::Iterative,
    ] {
        let ctx = simulated(1 << 30, 4, 1 << 40, pattern, None);
        let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
        let _: SKExecutionPlan = plan;
    }
}

/// Resolution happens per operation: fit and prediction with distinct contexts
/// produce independent plans.
#[test]
fn per_operation_resolution_produces_independent_plans() {
    // Fit: small in-memory dataset.
    let fit_ctx = simulated(1 << 30, 4, 100, SKAccessPattern::Sequential, None);
    // Prediction: volume exceeds memory.
    let pred_ctx = simulated(1 << 30, 4, 1 << 40, SKAccessPattern::RandomAccess, None);

    let fit_plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &fit_ctx).unwrap();
    let pred_plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &pred_ctx).unwrap();

    assert_eq!(fit_plan.mode, SKExecutionMode::InProcessSynchronous);
    assert_eq!(pred_plan.mode, SKExecutionMode::OutOfCoreMemoryMapped);
    assert_ne!(fit_plan, pred_plan);
}
