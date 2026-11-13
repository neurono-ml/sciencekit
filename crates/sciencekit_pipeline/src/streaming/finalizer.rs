//! The bounded batch finalizer buffer (spec `streaming-pipeline`, Decision 4).
//!
//! The batch finalizer needs the whole transformed dataset to call an existing
//! in-memory estimator's `fit`. The buffer is the explicit, bounded compromise:
//! it stores the transformed blocks until the accumulated resident size reaches
//! [`SKError::BatchBufferOversize`], and concatenates them into one block only at
//! the end. A true out-of-core (`partial_fit`) finalizer is future work.

use ndarray::{Array2, Axis};
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat};
use sprs::CsMat;

use super::memory::sk_charge_budget;

/// Accumulates transformed blocks within a memory budget.
pub(super) struct SKFinalizerBuffer<F: SKFloat> {
    blocks: Vec<SKFeatureBlock<F>>,
    used_bytes: usize,
    budget_bytes: usize,
    columns: Option<usize>,
}

impl<F: SKFloat> SKFinalizerBuffer<F> {
    /// Open an empty buffer with a hard resident-byte budget.
    pub(super) fn new(budget_bytes: usize) -> Self {
        SKFinalizerBuffer {
            blocks: Vec::new(),
            used_bytes: 0,
            budget_bytes,
            columns: None,
        }
    }

    /// Append one transformed block, enforcing a consistent column count.
    pub(super) fn push(&mut self, block: SKFeatureBlock<F>) -> Result<(), SKError> {
        let columns = block.number_of_columns();
        match self.columns {
            None => self.columns = Some(columns),
            Some(expected) if expected != columns => {
                return Err(SKError::shape_mismatch_2d(
                    block.number_of_rows(),
                    expected,
                    block.number_of_rows(),
                    columns,
                ));
            }
            Some(_) => {}
        }
        self.used_bytes = sk_charge_budget(self.used_bytes, block.byte_size(), self.budget_bytes)?;
        self.blocks.push(block);
        Ok(())
    }

    /// The resident bytes of the accumulated blocks.
    pub(super) fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// The accumulated column count, `None` when nothing was pushed.
    pub(super) fn columns(&self) -> Option<usize> {
        self.columns
    }

    /// Concatenate every block, preserving CSR when every block is sparse.
    pub(super) fn finish(self) -> Result<SKFeatureBlock<F>, SKError> {
        if self.blocks.is_empty() {
            return Err(SKError::EmptyInput {
                operation: "streaming_pipeline_finalizer",
            });
        }
        let any_sparse = self.blocks.iter().any(SKFeatureBlock::is_sparse);
        let all_sparse = self.blocks.iter().all(SKFeatureBlock::is_sparse);
        if all_sparse {
            return sk_concatenate_sparse(self.blocks);
        }
        if any_sparse {
            return sk_concatenate_dense_from_mixed(self.blocks);
        }
        sk_concatenate_dense(self.blocks)
    }
}

fn sk_concatenate_dense<F: SKFloat>(
    mut blocks: Vec<SKFeatureBlock<F>>,
) -> Result<SKFeatureBlock<F>, SKError> {
    if blocks.len() == 1 {
        return Ok(blocks.remove(0));
    }
    let views: Vec<_> = blocks
        .iter()
        .map(|block| match block {
            SKFeatureBlock::Dense(matrix) => matrix.view(),
            _ => unreachable!("the caller proved every block is dense"),
        })
        .collect();
    let matrix = ndarray::concatenate(Axis(0), &views)
        .map_err(|error| SKError::Conversion(format!("dense concatenation failed: {error}")))?;
    Ok(SKFeatureBlock::Dense(matrix))
}

fn sk_concatenate_sparse<F: SKFloat>(
    blocks: Vec<SKFeatureBlock<F>>,
) -> Result<SKFeatureBlock<F>, SKError> {
    let mut rows_count = 0;
    let mut columns_count = 0;
    let mut indptr = vec![0_usize];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for block in blocks {
        let SKFeatureBlock::Sparse(matrix) = block else {
            unreachable!("the caller proved every block is sparse");
        };
        rows_count += matrix.rows();
        columns_count = matrix.cols();
        for row in matrix.outer_iterator() {
            indices.extend(row.indices().iter().copied());
            data.extend(row.data().iter().copied());
            indptr.push(indices.len());
        }
    }
    let matrix = CsMat::new((rows_count, columns_count), indptr, indices, data);
    Ok(SKFeatureBlock::Sparse(matrix))
}

fn sk_concatenate_dense_from_mixed<F: SKFloat>(
    blocks: Vec<SKFeatureBlock<F>>,
) -> Result<SKFeatureBlock<F>, SKError> {
    let rows_count: usize = blocks.iter().map(SKFeatureBlock::number_of_rows).sum();
    let columns_count = blocks
        .first()
        .map(SKFeatureBlock::number_of_columns)
        .unwrap_or(0);
    let mut dense = Array2::zeros((rows_count, columns_count));
    let mut offset = 0;
    for block in &blocks {
        let rows = block.number_of_rows();
        match block {
            SKFeatureBlock::Dense(matrix) => {
                dense
                    .axis_iter_mut(Axis(0))
                    .skip(offset)
                    .zip(matrix.axis_iter(Axis(0)))
                    .for_each(|(mut target, source)| target.assign(&source));
            }
            SKFeatureBlock::Sparse(matrix) => {
                for (row_index, row) in matrix.outer_iterator().enumerate() {
                    for (&column, &value) in row.indices().iter().zip(row.data()) {
                        dense[[offset + row_index, column]] = value;
                    }
                }
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        }
        offset += rows;
    }
    Ok(SKFeatureBlock::Dense(dense))
}
