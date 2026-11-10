//! One streaming pass over a tabular source (spec `dataset-interop-core`).

use std::marker::PhantomData;

use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKLazySource, SKSupervisedBatch,
    SKSupervisedLazySource};

use super::SKTabularSource;
use crate::batch::SKFeatureLayout;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

/// One feature-only pass over a [`SKTabularSource`].
///
/// Each tabular batch is converted into an owned [`SKDataBatch`] under the
/// provider's selection, policy and layout. The source is borrowed, so the
/// provider can reopen it for another pass.
#[derive(Debug)]
pub struct SKTabularLazySource<'pass, Source, F> {
    pub(super) source: &'pass mut Source,
    pub(super) selection: &'pass SKColumnSelection,
    pub(super) policy: SKMissingValuePolicy,
    pub(super) layout: SKFeatureLayout,
    pub(super) marker: PhantomData<F>,
}

impl<F, Source> SKLazySource<F> for SKTabularLazySource<'_, Source, F>
where
    F: SKFloat,
    Source: SKTabularSource,
{
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        match self.source.next_batch()? {
            Some(batch) => Ok(Some(batch.to_data_batch::<F>(
                self.selection,
                self.policy,
                self.layout,
            )?)),
            None => Ok(None),
        }
    }
}

/// One supervised pass over a [`SKTabularSource`].
///
/// Each tabular batch is converted into an owned [`SKSupervisedBatch`] pairing
/// the selected features with the selected target.
#[derive(Debug)]
pub struct SKSupervisedTabularLazySource<'pass, Source, F> {
    pub(super) source: &'pass mut Source,
    pub(super) selection: &'pass SKColumnSelection,
    pub(super) policy: SKMissingValuePolicy,
    pub(super) layout: SKFeatureLayout,
    pub(super) marker: PhantomData<F>,
}

impl<F, Source> SKSupervisedLazySource<F> for SKSupervisedTabularLazySource<'_, Source, F>
where
    F: SKFloat,
    Source: SKTabularSource,
{
    type Error = SKError;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error> {
        match self.source.next_batch()? {
            Some(batch) => Ok(Some(batch.to_supervised_batch::<F>(
                self.selection,
                self.policy,
                self.layout,
            )?)),
            None => Ok(None),
        }
    }
}
