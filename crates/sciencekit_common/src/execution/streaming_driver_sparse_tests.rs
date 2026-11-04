//! Tests for sparse blocks through the streaming driver (spec
//! `sparse-streaming-batches`).

use sprs::CsMat;

use super::streaming_driver::SKStreamDecision;
use super::{SKExecutionMode, SKExecutionPlan, sk_run_streaming_driver};
use crate::{SKDataBatch, SKDataView, SKError, SKLazySource};

/// A source of sparse batches: each block has shape `1 × 4` with two
/// non-zeros that encode its position.
struct SparseRecordingSource {
    total: usize,
    next: usize,
}

impl SKLazySource<f64> for SparseRecordingSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if self.next >= self.total {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let block = sparse_row(position);
        Ok(Some(SKDataBatch::new_sparse(
            block,
            position,
            position == self.total - 1,
        )))
    }
}

fn sparse_row(position: usize) -> CsMat<f64> {
    CsMat::new(
        (1, 4),
        vec![0, 2],
        vec![1, 3],
        vec![position as f64, (position * 10) as f64],
    )
}

fn streaming_plan() -> SKExecutionPlan {
    SKExecutionPlan {
        mode: SKExecutionMode::OutOfCoreStreaming,
        parallelism: 2,
        batch_size: Some(2),
        buffer_depth: 1,
        available_memory_bytes: u64::MAX,
    }
}

/// The driver hands a sparse block to the update closure unchanged: same
/// representation, same stored non-zeros, in source order.
#[test]
fn sparse_block_reaches_the_update_closure_unchanged() {
    let mut source = SparseRecordingSource { total: 4, next: 0 };
    let update =
        |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<(usize, Vec<f64>)>| {
            assert!(batch.is_sparse());
            assert_eq!(batch.number_of_rows(), 1);
            assert_eq!(batch.number_of_columns(), 4);
            match batch.view() {
                SKDataView::Sparse(view) => {
                    assert_eq!(view.nnz(), 2);
                    state.push((batch.position(), view.data().to_vec()));
                }
                SKDataView::Dense(_) => panic!("the driver must not densify a sparse block"),
            }
            SKStreamDecision::Continue
        };
    let mut state = Vec::new();
    sk_run_streaming_driver(&mut source, &mut state, update, &streaming_plan()).unwrap();

    assert_eq!(
        state,
        vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 10.0]),
            (2, vec![2.0, 20.0]),
            (3, vec![3.0, 30.0]),
        ]
    );
}

/// With a deeper buffer and more than one compute thread, block order and the
/// sparse representation are preserved.
#[test]
fn deep_buffer_and_parallelism_preserve_order_and_representation() {
    let plan = SKExecutionPlan {
        parallelism: 4,
        buffer_depth: 2,
        ..streaming_plan()
    };
    let mut source = SparseRecordingSource { total: 12, next: 0 };
    let mut state = Vec::new();
    let update = |batch: SKDataBatch<f64>, _parallelism: usize, state: &mut Vec<(usize, bool)>| {
        assert!(batch.is_sparse());
        state.push((batch.position(), batch.is_sparse()));
        SKStreamDecision::Continue
    };
    sk_run_streaming_driver(&mut source, &mut state, update, &plan).unwrap();
    assert_eq!(state.len(), 12);
    for (expected, (position, is_sparse)) in state.iter().enumerate() {
        assert_eq!(*position, expected);
        assert!(*is_sparse);
    }
}
