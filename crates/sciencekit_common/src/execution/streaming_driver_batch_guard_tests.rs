//! Guard tests for sparse blocks through the streaming driver (spec
//! `sparse-streaming-batches`).

use ndarray::Array2;
use sprs::CsMat;

use super::streaming_driver::SKStreamDecision;
use super::{SKExecutionMode, SKExecutionPlan, sk_run_streaming_driver};
use crate::{SKBackendKind, SKDataBatch, SKError, SKFeatureBlock, SKLazySource};

/// A source of owned batches in a fixed order.
struct BlockSource {
    batches: Vec<SKDataBatch<f64>>,
    next: usize,
}

impl SKLazySource<f64> for BlockSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let batch = self.batches[self.next].clone();
        self.next += 1;
        Ok(Some(batch))
    }
}

fn guard_plan(available_memory_bytes: u64) -> SKExecutionPlan {
    SKExecutionPlan {
        mode: SKExecutionMode::OutOfCoreStreaming,
        parallelism: 2,
        batch_size: Some(2),
        buffer_depth: 1,
        available_memory_bytes,
        backend: SKBackendKind::Faer,
        fallback_reason: None,
    }
}

/// A 1000 × 1000 sparse block with two non-zeros: its dense figure is 8 MB,
/// its stored CSR arrays are a few kilobytes.
fn sparse_block() -> SKFeatureBlock<f64> {
    SKFeatureBlock::Sparse(sparse_matrix())
}

fn sparse_matrix() -> CsMat<f64> {
    let mut indptr = vec![1_usize; 1001];
    indptr[0] = 0;
    indptr[1000] = 2;
    CsMat::new((1000, 1000), indptr, vec![1, 999], vec![1.0_f64, 2.0])
}

/// The guard checks the batch's stored size when the batch is read: a sparse
/// block that fits twice is accepted, while a dense block of the same shape is
/// rejected before the update closure sees it.
#[test]
fn guard_uses_the_sparse_byte_size_and_rejects_a_dense_batch() {
    let sparse = sparse_block();
    let dense_bytes = 1000 * 1000 * size_of::<f64>();
    let available = 2 * sparse.byte_size() as u64 + 1;
    assert!(available < 2 * dense_bytes as u64);
    let plan = guard_plan(available);

    let mut sparse_source = BlockSource {
        batches: vec![SKDataBatch::new_sparse(sparse_matrix(), 0, true)],
        next: 0,
    };
    let mut seen_sparse = false;
    sk_run_streaming_driver(
        &mut sparse_source,
        &mut seen_sparse,
        |batch: SKDataBatch<f64>, _, seen: &mut bool| {
            *seen = batch.is_sparse();
            SKStreamDecision::Continue
        },
        &plan,
    )
    .expect("a sparse block that fits twice is accepted");
    assert!(seen_sparse);

    let mut dense_source = BlockSource {
        batches: vec![SKDataBatch::new_dense(
            Array2::<f64>::zeros((1000, 1000)),
            0,
            true,
        )],
        next: 0,
    };
    let mut seen_dense = false;
    let error = sk_run_streaming_driver(
        &mut dense_source,
        &mut seen_dense,
        |_batch: SKDataBatch<f64>, _, seen: &mut bool| {
            *seen = true;
            SKStreamDecision::Continue
        },
        &plan,
    )
    .expect_err("a dense block of the same shape does not fit");
    assert!(matches!(error, SKError::BatchBufferOversize { .. }));
    assert!(
        !seen_dense,
        "the guard must reject the batch before the update"
    );
}
