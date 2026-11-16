//! Tests for the access-pattern vocabulary and its parity tier (spec
//! `access-pattern`).

use super::{SKAccessPattern, SKExecutionContext, SKExecutionMode, sk_resolve_execution_plan};

fn context(pattern: SKAccessPattern, dataset_bytes: u64) -> SKExecutionContext {
    SKExecutionContext {
        available_memory_bytes: 1 << 20,
        cpu_cores: 4,
        dataset_size_bytes: dataset_bytes,
        dataset_elements: Some(64),
        scalar_size_bytes: 8,
        grain: 1024,
        access_pattern: pattern,
        batch_size_hint: Some(16),
        batch_byte_size_hint: None,
        device: None,
    }
}

#[test]
fn the_vocabulary_has_exactly_four_values() {
    let patterns = [
        SKAccessPattern::Sequential,
        SKAccessPattern::RandomAccess,
        SKAccessPattern::Iterative,
        SKAccessPattern::FullMemory,
    ];
    assert_eq!(patterns.len(), 4);
    for (index, left) in patterns.iter().enumerate() {
        for right in &patterns[index + 1..] {
            assert_ne!(left, right);
        }
    }
}

#[test]
fn every_pattern_maps_to_its_parity_tier() {
    assert_eq!(SKAccessPattern::Sequential.tier(), "streaming");
    assert_eq!(SKAccessPattern::RandomAccess.tier(), "mappable");
    assert_eq!(SKAccessPattern::Iterative.tier(), "chunked");
    assert_eq!(SKAccessPattern::FullMemory.tier(), "full-memory");
}

#[test]
fn automatic_plan_keeps_a_full_memory_estimator_in_memory() {
    let oversized = context(SKAccessPattern::FullMemory, 1 << 40);
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &oversized).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::InProcessSynchronous);
}
