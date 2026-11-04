//! The single-pass logistic stream accumulator: every batch's design and its
//! integer-coded labels, dense or CSR.
//!
//! When any block is sparse the retained design is upgraded to CSR; dense blocks
//! are converted once at solve time, never densifying a sparse block.

use ndarray::ArrayView2;
use sciencekit_common::execution::SKStreamDecision;
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat, SKSupervisedBatch};
use sprs::{CsMat, TriMat};

use super::fit::sk_float_label;
use crate::logistic::SKLogisticState;

/// The single-pass accumulator: every batch's design and integer-coded labels.
pub(super) struct SKLogisticStreamAccumulator<F> {
    pub(super) batches: Vec<SKFeatureBlock<F>>,
    pub(super) labels: Vec<String>,
    pub(super) columns_count: usize,
    pub(super) error: Option<SKError>,
}

impl<F: SKFloat> SKLogisticStreamAccumulator<F> {
    pub(super) fn new() -> Self {
        Self {
            batches: Vec::new(),
            labels: Vec::new(),
            columns_count: 0,
            error: None,
        }
    }

    pub(super) fn update(&mut self, batch: SKSupervisedBatch<F>) -> SKStreamDecision {
        match self.accumulate(batch) {
            Ok(()) => SKStreamDecision::Continue,
            Err(error) => {
                self.error = Some(error);
                SKStreamDecision::Stop
            }
        }
    }

    fn accumulate(&mut self, batch: SKSupervisedBatch<F>) -> Result<(), SKError> {
        let rows_count = batch.number_of_rows();
        let columns_count = batch.number_of_columns();
        if rows_count == 0 {
            return Ok(());
        }
        if self.columns_count != 0 && self.columns_count != columns_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.columns_count],
                found: vec![columns_count],
            });
        }
        self.columns_count = columns_count;
        let targets = batch.targets();
        if targets.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![targets.len()],
            });
        }
        targets
            .iter()
            .for_each(|value| self.labels.push(sk_float_label(*value)));
        self.batches.push(match batch.view() {
            sciencekit_common::SKDataView::Dense(features) => {
                SKFeatureBlock::Dense(features.to_owned())
            }
            sciencekit_common::SKDataView::Sparse(features) => {
                SKFeatureBlock::Sparse(features.to_owned())
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        });
        Ok(())
    }

    /// Accumulate the retained design into the matching state at solve time.
    pub(super) fn into_state(self, canonical_labels: &[usize]) -> SKLogisticState<F> {
        let any_sparse = self.batches.iter().any(SKFeatureBlock::is_sparse);
        let mut state = if any_sparse {
            SKLogisticState::new_sparse()
        } else {
            SKLogisticState::new_dense()
        };
        let mut offset = 0;
        for block in &self.batches {
            let rows = block.number_of_rows();
            let labels = &canonical_labels[offset..offset + rows];
            match block {
                SKFeatureBlock::Dense(features) => {
                    if any_sparse {
                        let csr = sk_dense_to_csr(&features.view());
                        state.accumulate_sparse(csr.view(), labels);
                    } else {
                        state.accumulate_dense(features.view(), labels);
                    }
                }
                SKFeatureBlock::Sparse(features) => {
                    state.accumulate_sparse(features.view(), labels)
                }
                _ => unreachable!("the block carries one of two representations"),
            }
            offset += rows;
        }
        state
    }
}

/// Convert a dense design to CSR, dropping exact zeros.
fn sk_dense_to_csr<F: SKFloat>(features: &ArrayView2<'_, F>) -> CsMat<F> {
    let mut triplets = TriMat::new((features.nrows(), features.ncols()));
    features.indexed_iter().for_each(|((row, column), &value)| {
        if value != F::zero() {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}
