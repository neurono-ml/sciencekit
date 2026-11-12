//! Chained fitting helpers: block rebatching, stage application and the
//! bounded materialization of a one-shot provider.
//!
//! A one-shot provider cannot be reopened for the per-stage passes, so the
//! pipeline reads it once into memory — bounded by the same budget as the
//! finalizer — and then serves the materialized blocks through a reopenable
//! in-memory provider. This is the documented alternative the spec allows in
//! place of a structured reopenable error.

use sciencekit_common::{
    SKDataBatch, SKError, SKFeatureBlock, SKFloat, SKLazySource, SKSourceProvider,
};

use super::memory::sk_charge_budget;
use super::stage::SKFittedStage;

/// Rebuild a batch from an owned block, preserving dense/CSR.
pub(super) fn sk_data_batch_from_block<F: SKFloat>(
    block: SKFeatureBlock<F>,
    position: usize,
    is_final: bool,
) -> Result<SKDataBatch<F>, SKError> {
    match block {
        SKFeatureBlock::Dense(matrix) => Ok(SKDataBatch::new_dense(matrix, position, is_final)),
        SKFeatureBlock::Sparse(matrix) => Ok(SKDataBatch::new_sparse(matrix, position, is_final)),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }),
    }
}

/// Run an owned block through every fitted stage in order.
pub(super) fn sk_apply_fitted_stages<F: SKFloat>(
    stages: &[Box<dyn SKFittedStage<F>>],
    block: SKFeatureBlock<F>,
) -> Result<SKFeatureBlock<F>, SKError> {
    let mut current = block;
    for (index, stage) in stages.iter().enumerate() {
        current = stage
            .transform_stage_batch(&current)
            .map_err(|error| super::errors::sk_stage_failure(index, error))?;
    }
    Ok(current)
}

/// Materialize a one-shot provider within the memory budget.
///
/// Returns the reopenable in-memory provider and the resident bytes it holds.
pub(super) fn sk_materialize<F, Provider>(
    provider: &mut Provider,
    budget_bytes: usize,
) -> Result<(SKMaterializedProvider<F>, usize), SKError>
where
    F: SKFloat,
    Provider: SKSourceProvider<F>,
{
    let mut source = provider.open()?;
    let mut blocks = Vec::new();
    let mut used_bytes = 0;
    while let Some(batch) = source.next_batch().map_err(Into::into)? {
        let block = SKFeatureBlock::from(batch.view());
        used_bytes = sk_charge_budget(used_bytes, block.byte_size(), budget_bytes)?;
        blocks.push(block);
    }
    Ok((SKMaterializedProvider { blocks }, used_bytes))
}

/// A reopenable provider over blocks held in memory.
pub(super) struct SKMaterializedProvider<F> {
    blocks: Vec<SKFeatureBlock<F>>,
}

impl<F: SKFloat> SKSourceProvider<F> for SKMaterializedProvider<F> {
    type Error = SKError;

    type Source<'pass>
        = SKMaterializedSource<'pass, F>
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        Ok(SKMaterializedSource {
            blocks: &self.blocks,
            cursor: 0,
        })
    }
}

/// One pass over the materialized blocks.
pub(super) struct SKMaterializedSource<'blocks, F: SKFloat> {
    blocks: &'blocks [SKFeatureBlock<F>],
    cursor: usize,
}

impl<F: SKFloat> SKLazySource<F> for SKMaterializedSource<'_, F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        let Some(block) = self.blocks.get(self.cursor) else {
            return Ok(None);
        };
        let position = self.cursor;
        self.cursor += 1;
        sk_data_batch_from_block(block.clone(), position, self.cursor == self.blocks.len())
            .map(Some)
    }
}
