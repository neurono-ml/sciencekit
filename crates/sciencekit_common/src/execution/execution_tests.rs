//! Tests for execution planning (spec `execution-planning`).

use super::{
    SKAccessPattern, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    sk_resolve_execution_plan,
};
use crate::SKError;

/// Provisional per-kernel grain (elements per thread). Replaced by the
/// calibration values recorded in task 3.2.
/// Per-kernel grain injected into the simulated context.
const GRAIN: usize = 1024;

/// Build a simulated context (never reads the physical machine).
#[allow(clippy::too_many_arguments)]
fn simulated(
    memory: u64,
    cores: usize,
    dataset_bytes: u64,
    dataset_elements: Option<u64>,
    scalar_size: u64,
    pattern: SKAccessPattern,
    batch: Option<usize>,
) -> SKExecutionContext {
    SKExecutionContext {
        available_memory_bytes: memory,
        cpu_cores: cores,
        dataset_size_bytes: dataset_bytes,
        dataset_elements,
        scalar_size_bytes: scalar_size,
        grain: GRAIN,
        access_pattern: pattern,
        batch_size_hint: batch,
    }
}

/// Automatic intent over a known context produces a concrete plan.
#[test]
fn automatic_intent_produces_concrete_plan() {
    // Dataset (100 elements) fits in memory (1 GiB) but is below one grain.
    let ctx = simulated(
        1 << 30,
        8,
        100,
        Some(100),
        8,
        SKAccessPattern::Sequential,
        None,
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::InProcessSynchronous);
    assert_eq!(plan.parallelism, 1);
}

/// Automatic intent over a larger-than-memory sequential dataset streams.
#[test]
fn automatic_oversized_sequential_streams() {
    let ctx = simulated(
        1 << 30,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(1024),
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.batch_size, Some(1024));
}

/// Automatic intent over a larger-than-memory random-access dataset maps.
#[test]
fn automatic_oversized_random_maps() {
    let ctx = simulated(
        1 << 30,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::RandomAccess,
        None,
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreMemoryMapped);
    // Memory-mapped shards run independently: parallelism equals the core count.
    assert_eq!(plan.parallelism, 4);
}

/// Explicit intent compatible with the pattern is preserved in the plan.
#[test]
fn explicit_compatible_intent_is_preserved() {
    let ctx = simulated(
        1 << 30,
        2,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        None,
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::OutOfCoreStreaming, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
}

/// Same intent + context → identical plans (deterministic, pure).
#[test]
fn same_context_produces_same_plan() {
    let ctx = simulated(
        1 << 30,
        6,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(64),
    );
    let a = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    let b = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(a, b);
}

/// Simulated context drives resolution without reading the physical machine.
#[test]
fn simulated_context_dispenses_with_real_machine() {
    let ctx = simulated(1, 1, 10, None, 8, SKAccessPattern::Sequential, None);
    // With only 1 byte of memory and a 10-byte dataset, automatic must stream.
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.parallelism, 1);
}

/// Explicit streaming on a random-access algorithm fails with a precise error.
#[test]
fn sequential_streaming_refused_for_random_access() {
    let ctx = simulated(
        1 << 30,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::RandomAccess,
        None,
    );
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
        let ctx = simulated(1 << 30, 4, 1 << 40, None, 8, pattern, None);
        let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
        let _: SKExecutionPlan = plan;
    }
}

/// Resolution happens per operation: fit and prediction with distinct contexts
/// produce independent plans.
#[test]
fn per_operation_resolution_produces_independent_plans() {
    // Fit: small in-memory dataset.
    let fit_ctx = simulated(
        1 << 30,
        4,
        100,
        Some(100),
        8,
        SKAccessPattern::Sequential,
        None,
    );
    // Prediction: volume exceeds memory.
    let pred_ctx = simulated(
        1 << 30,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::RandomAccess,
        None,
    );

    let fit_plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &fit_ctx).unwrap();
    let pred_plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &pred_ctx).unwrap();

    assert_eq!(fit_plan.mode, SKExecutionMode::InProcessSynchronous);
    assert_eq!(pred_plan.mode, SKExecutionMode::OutOfCoreMemoryMapped);
    assert_ne!(fit_plan, pred_plan);
}

// ---- Task 1.1 scenarios (spec `execution-planning`, size-aware resolution) ----

/// An in-memory dataset below one grain of elements per thread resolves to
/// exactly one thread (sequential, no dispatch overhead).
#[test]
fn small_in_memory_dataset_resolves_sequential() {
    let ctx = simulated(
        1 << 30,
        8,
        100,
        Some(100),
        8,
        SKAccessPattern::Sequential,
        None,
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::InProcessSynchronous);
    assert_eq!(plan.parallelism, 1);
}

/// A large in-memory dataset parallelizes up to the core count, capped by the
/// grain-derived thread estimate.
#[test]
fn large_in_memory_dataset_parallelizes_up_to_cores() {
    let ctx = simulated(
        1 << 30,
        8,
        1_000_000 * 8,
        Some(1_000_000),
        8,
        SKAccessPattern::Sequential,
        None,
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::InProcessSynchronous);
    // ceil(1_000_000 / 1024) = 977 → capped at 8 cores.
    assert_eq!(plan.parallelism, 8);
}

/// Streaming resolves against the batch as the work unit: a batch below one
/// grain stays sequential even though the whole dataset is larger than memory.
#[test]
fn streaming_batch_below_grain_resolves_sequential() {
    let ctx = simulated(
        1 << 30,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(500),
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.parallelism, 1);
}

/// A streaming batch above one grain resolves to the core count.
#[test]
fn streaming_batch_above_grain_parallelizes_up_to_cores() {
    let ctx = simulated(
        1 << 30,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(10_000),
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    // ceil(10_000 / 1024) = 10 → capped at 4 cores.
    assert_eq!(plan.parallelism, 4);
}

/// Memory-mapped access resolves full sharding: parallelism equals the cores.
#[test]
fn memory_mapped_resolves_full_sharding() {
    let ctx = simulated(
        1 << 30,
        8,
        1 << 40,
        None,
        8,
        SKAccessPattern::RandomAccess,
        None,
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreMemoryMapped);
    assert_eq!(plan.parallelism, 8);
}

/// A single-core machine always resolves to parallelism one.
#[test]
fn single_core_machine_resolves_sequential_everywhere() {
    for pattern in [
        SKAccessPattern::Sequential,
        SKAccessPattern::RandomAccess,
    ] {
        let ctx = simulated(
            1 << 30,
            1,
            1 << 40,
            None,
            8,
            pattern,
            Some(100_000),
        );
        let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
        assert_eq!(plan.parallelism, 1);
    }
}

// ---- Task 1.3 scenarios (double-buffer memory guard, spec `execution-planning`) ----

/// An oversized batch hint is refused before any processing, with a structured
/// error naming the batch size and the available memory.
#[test]
fn oversized_batch_hint_is_refused_before_processing() {
    // 1 MiB available; a 1 MiB-element batch × 8 bytes = 8 MiB per resident
    // batch; double buffering needs 16 MiB — over the budget.
    let ctx = simulated(
        1 << 20,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(1 << 20),
    );
    match sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx) {
        Err(SKError::BatchBufferOversize {
            batch_bytes,
            required_bytes,
            available_bytes,
        }) => {
            assert_eq!(batch_bytes, 1 << 23);
            assert_eq!(required_bytes, 1 << 24);
            assert_eq!(available_bytes, 1 << 20);
        }
        other => panic!("expected oversize error, got {other:?}"),
    }
}

/// The same guard fires for an explicit streaming intent.
#[test]
fn explicit_streaming_refuses_oversized_batch() {
    let ctx = simulated(
        1 << 20,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(1 << 20),
    );
    match sk_resolve_execution_plan(SKExecutionMode::OutOfCoreStreaming, &ctx) {
        Err(SKError::BatchBufferOversize { .. }) => {}
        other => panic!("expected oversize error, got {other:?}"),
    }
}

/// A batch that just fits the double-buffer budget passes the guard.
#[test]
fn batch_fitting_double_buffer_budget_passes() {
    // 16 MiB available; 1 MiB-element batch × 8 = 8 MiB; 2 × 8 MiB = 16 MiB == budget.
    let ctx = simulated(
        1 << 24,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(1 << 20),
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.batch_size, Some(1 << 20));
}

/// Automatic intent never trips the guard when the resolved batch fits.
#[test]
fn automatic_intent_never_trips_guard_when_batch_fits() {
    let ctx = simulated(
        1 << 30,
        4,
        1 << 40,
        None,
        8,
        SKAccessPattern::Sequential,
        Some(1024),
    );
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &ctx).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
    assert_eq!(plan.batch_size, Some(1024));
}