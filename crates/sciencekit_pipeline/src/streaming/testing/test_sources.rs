//! Test-only streaming sources and block builders.

use ndarray::Array2;
use sciencekit_common::{
    SKDataBatch, SKError, SKFeatureBlock, SKFloat, SKLazySource, SKReopenableSource,
};

/// A finite source over owned blocks that can fail at a chosen batch.
pub(crate) struct SKTestBlocksSource<F> {
    blocks: Vec<SKFeatureBlock<F>>,
    cursor: usize,
    fail_at: Option<usize>,
}

impl<F> SKTestBlocksSource<F> {
    pub(crate) fn new(blocks: Vec<SKFeatureBlock<F>>) -> Self {
        SKTestBlocksSource {
            blocks,
            cursor: 0,
            fail_at: None,
        }
    }

    pub(crate) fn new_failing(blocks: Vec<SKFeatureBlock<F>>, fail_at: usize) -> Self {
        SKTestBlocksSource {
            blocks,
            cursor: 0,
            fail_at: Some(fail_at),
        }
    }
}

impl<F: SKFloat> SKLazySource<F> for SKTestBlocksSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        if self.fail_at == Some(self.cursor) {
            return Err(SKError::Conversion(format!(
                "intentional source failure at batch {}",
                self.cursor
            )));
        }
        let Some(block) = self.blocks.get(self.cursor) else {
            return Ok(None);
        };
        let position = self.cursor;
        self.cursor += 1;
        let is_final = self.cursor == self.blocks.len();
        Ok(Some(match (*block).clone() {
            SKFeatureBlock::Dense(matrix) => SKDataBatch::new_dense(matrix, position, is_final),
            SKFeatureBlock::Sparse(matrix) => SKDataBatch::new_sparse(matrix, position, is_final),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        }))
    }
}

/// A reopenable provider over owned blocks.
pub(crate) fn sk_reopenable_blocks<F: SKFloat>(
    blocks: Vec<SKFeatureBlock<F>>,
) -> SKReopenableSource<impl FnMut() -> Result<SKTestBlocksSource<F>, SKError>> {
    SKReopenableSource::new(move || Ok(SKTestBlocksSource::new(blocks.clone())))
}

/// One dense block.
pub(crate) fn sk_single_block<F: SKFloat>(
    rows: usize,
    columns: usize,
    values: Vec<F>,
) -> SKFeatureBlock<F> {
    SKFeatureBlock::Dense(Array2::from_shape_vec((rows, columns), values).unwrap())
}

/// Three hand-checkable 2×2 dense blocks: rows 1..=12 in order.
pub(crate) fn sk_tiny_blocks<F: SKFloat>() -> Vec<SKFeatureBlock<F>> {
    let values: Vec<F> = (1..=12)
        .map(|value| F::from(value).expect("a small integer converts to the scalar"))
        .collect();
    values
        .chunks(4)
        .map(|chunk| sk_single_block(2, 2, chunk.to_vec()))
        .collect()
}

/// A reopenable provider whose every pass fails at the chosen batch.
pub(crate) fn sk_reopenable_failing_blocks<F: SKFloat>(
    blocks: Vec<SKFeatureBlock<F>>,
    fail_at: usize,
) -> SKReopenableSource<impl FnMut() -> Result<SKTestBlocksSource<F>, SKError>> {
    SKReopenableSource::new(move || Ok(SKTestBlocksSource::new_failing(blocks.clone(), fail_at)))
}
