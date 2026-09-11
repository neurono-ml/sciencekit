//! Tests for the measured grain constants (spec `higher-order-kernels`,
//! provenance scenario).

use super::{
    SK_AXIS_SUM_GRAIN, SK_BINARY_COMBINE_GRAIN, SK_ELEMENTWISE_TRANSFORM_GRAIN,
    SK_SCALE_IN_PLACE_GRAIN,
};
use sciencekit_common::execution::{
    SKAccessPattern, SKExecutionContext, SKExecutionMode, sk_resolve_execution_plan,
};

/// Every kernel exposes a positive grain constant (the provenance and measured
/// value are documented beside each constant in `grain.rs`).
#[test]
fn every_kernel_exposes_a_positive_grain() {
    for grain in [
        SK_ELEMENTWISE_TRANSFORM_GRAIN,
        SK_BINARY_COMBINE_GRAIN,
        SK_AXIS_SUM_GRAIN,
        SK_SCALE_IN_PLACE_GRAIN,
    ] {
        assert!(grain > 0);
    }
}

/// The same grain constant feeds resolution unchanged: a unit below the grain
/// stays sequential, a unit far above it parallelizes.
#[test]
fn grain_governs_resolution_across_regimes() {
    let grain = SK_AXIS_SUM_GRAIN;
    let below = context(Some((grain - 1) as u64), grain, 8);
    let above = context(Some((grain * 8) as u64), grain, 8);

    let below_plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &below).unwrap();
    let above_plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &above).unwrap();

    assert_eq!(below_plan.parallelism, 1);
    assert_eq!(above_plan.parallelism, 8);
}

/// Build an in-memory simulated context carrying a kernel grain.
fn context(dataset_elements: Option<u64>, grain: usize, cores: usize) -> SKExecutionContext {
    SKExecutionContext {
        available_memory_bytes: 1 << 40,
        cpu_cores: cores,
        dataset_size_bytes: 1,
        dataset_elements,
        scalar_size_bytes: 8,
        grain,
        access_pattern: SKAccessPattern::Sequential,
        batch_size_hint: None,
    }
}