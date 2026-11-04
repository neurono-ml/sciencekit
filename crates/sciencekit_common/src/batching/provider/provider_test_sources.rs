//! Shared fixtures for the source provider tests.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::batching::{SKDataBatch, SKLazySource, SKSupervisedBatch, SKSupervisedLazySource};
use crate::{SKError, SKFloat};

/// A finite source of `total` one-row batches; usable as both families.
#[derive(Debug)]
pub(super) struct FiniteSource<F> {
    next: usize,
    total: usize,
    marker: PhantomData<F>,
}

impl<F: SKFloat> FiniteSource<F> {
    pub(super) fn new(total: usize) -> Self {
        FiniteSource {
            next: 0,
            total,
            marker: PhantomData,
        }
    }
}

impl<F: SKFloat> SKLazySource<F> for FiniteSource<F> {
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

impl<F: SKFloat> SKSupervisedLazySource<F> for FiniteSource<F> {
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

pub(super) fn drain_feature_positions<F: SKFloat>(source: &mut FiniteSource<F>) -> Vec<usize> {
    let mut positions = Vec::new();
    while let Some(batch) = source.next_batch().unwrap() {
        positions.push(batch.position());
    }
    positions
}

pub(super) fn drain_supervised_positions<F: SKFloat>(source: &mut FiniteSource<F>) -> Vec<usize> {
    let mut positions = Vec::new();
    while let Some(batch) = source.next_supervised_batch().unwrap() {
        positions.push(batch.position());
    }
    positions
}
