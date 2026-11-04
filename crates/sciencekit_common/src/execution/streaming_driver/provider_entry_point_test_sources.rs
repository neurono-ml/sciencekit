//! Shared fixtures for the provider entry point tests.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::SKError;
use crate::SKFloat;
use crate::batching::{SKDataBatch, SKLazySource, SKSupervisedBatch, SKSupervisedLazySource};
use crate::execution::{SKExecutionMode, SKExecutionPlan};

/// A finite source of `total` one-row batches; recording source order.
#[derive(Debug)]
pub(super) struct RecordingSource<F> {
    next: usize,
    total: usize,
    marker: PhantomData<F>,
}

impl<F: SKFloat> RecordingSource<F> {
    pub(super) fn new(total: usize) -> Self {
        RecordingSource {
            next: 0,
            total,
            marker: PhantomData,
        }
    }
}

impl<F: SKFloat> SKLazySource<F> for RecordingSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        if self.next >= self.total {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let data = Array2::from_shape_vec((1, 1), vec![F::from(position as f64).unwrap()]).unwrap();
        Ok(Some(SKDataBatch::new(
            data,
            position,
            position == self.total - 1,
        )))
    }
}

impl<F: SKFloat> SKSupervisedLazySource<F> for RecordingSource<F> {
    type Error = SKError;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error> {
        if self.next >= self.total {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let features =
            Array2::from_shape_vec((1, 1), vec![F::from(position as f64).unwrap()]).unwrap();
        let targets = Array1::from_vec(vec![F::from((position * 10) as f64).unwrap()]);
        Ok(Some(SKSupervisedBatch::new(
            features,
            targets,
            position,
            position == self.total - 1,
        )))
    }
}

pub(super) fn streaming_plan() -> SKExecutionPlan {
    SKExecutionPlan {
        mode: SKExecutionMode::OutOfCoreStreaming,
        parallelism: 2,
        batch_size: Some(1),
        buffer_depth: 1,
    }
}
