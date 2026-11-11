//! Tests for sparse memory accounting in the streaming buffer guard (spec
//! `sparse-streaming-batches`).

use sprs::CsMat;

use super::{SKAccessPattern, SKExecutionContext, SKExecutionMode, sk_resolve_execution_plan};
use crate::{SKError, SKFeatureBlock};

const GRAIN: usize = 1024;

/// A 1000 × 1000 sparse block with two non-zeros: its dense figure is 8 MB,
/// its stored CSR arrays are a few kilobytes.
fn sparse_block() -> SKFeatureBlock<f64> {
    let mut indptr = vec![1_usize; 1001];
    indptr[0] = 0;
    indptr[1000] = 2;
    SKFeatureBlock::Sparse(CsMat::new(
        (1000, 1000),
        indptr,
        vec![1, 999],
        vec![1.0_f64, 2.0],
    ))
}

fn simulated(
    memory: u64,
    batch_elements: usize,
    batch_byte_size: Option<u64>,
) -> SKExecutionContext {
    SKExecutionContext {
        available_memory_bytes: memory,
        cpu_cores: 2,
        dataset_size_bytes: 1 << 40,
        dataset_elements: None,
        scalar_size_bytes: 8,
        grain: GRAIN,
        access_pattern: SKAccessPattern::Sequential,
        batch_size_hint: Some(batch_elements),
        batch_byte_size_hint: batch_byte_size,
        device: None,
    }
}

/// The doubled size checked against memory is the stored sparse size, so a
/// sparse hint that fits is accepted even though the dense figure would not.
#[test]
fn guard_uses_the_sparse_byte_size() {
    let block = sparse_block();
    let dense_figure = 1000 * 1000 * size_of::<f64>();
    // Two sparse buffers fit; two dense figures do not.
    let available = 2 * block.byte_size() as u64 + 1;
    assert!(available < 2 * dense_figure as u64);

    let context = simulated(available, 1000 * 1000, Some(block.byte_size() as u64));
    let plan = sk_resolve_execution_plan(SKExecutionMode::OutOfCoreStreaming, &context).unwrap();
    assert_eq!(plan.mode, SKExecutionMode::OutOfCoreStreaming);
}

/// Without the byte-size hint, the guard falls back to the dense element
/// figure and rejects the same shape — proving the hint is what changed.
#[test]
fn dense_fallback_uses_rows_times_columns() {
    let block = sparse_block();
    let available = 2 * block.byte_size() as u64 + 1;
    let context = simulated(available, 1000 * 1000, None);
    match sk_resolve_execution_plan(SKExecutionMode::OutOfCoreStreaming, &context) {
        Err(SKError::BatchBufferOversize { .. }) => {}
        other => panic!("expected the dense figure to overflow, got {other:?}"),
    }
}
