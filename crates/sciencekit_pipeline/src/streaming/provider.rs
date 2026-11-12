//! Source-provider adapters for boxed streaming stages.
//!
//! Two adapters live here:
//!
//! - [`SKStreamingProvider`] erases a concrete [`SKSourceProvider`] into an
//!   object-safe view so the pipeline can store heterogeneous stages.
//! - [`SKTransformingProvider`] wraps an upstream provider and pushes every
//!   batch through the stages that were already fitted, which is how a stage
//!   is fitted on the previous stage's transformed output.

use sciencekit_common::{
    SKDataBatch, SKError, SKFeatureBlock, SKFloat, SKLazySource, SKSourceProvider,
};

use super::stage::SKFittedStage;

/// An object-safe view of a source provider.
///
/// The generic [`SKSourceProvider`] cannot be stored behind `dyn` because it
/// has a generic associated source type. This trait erases that type: every
/// pass yields a boxed [`SKLazySource`] whose error is already the central
/// [`SKError`]. The pipeline only calls it after materializing a one-shot
/// provider, so every open source can be reopened.
pub(super) trait SKStreamingProvider<F: SKFloat> {
    /// Open one pass as a boxed source.
    fn open_object_source(&mut self) -> Result<SKBoxedSource<'_, F>, SKError>;
}

/// A boxed source that implements [`SKLazySource`] locally.
///
/// The orphan rule forbids implementing the shared trait for the standard
/// library's `Box`, so this newtype carries the boxed object and forwards every
/// read.
pub(super) struct SKBoxedSource<'pass, F: SKFloat> {
    inner: Box<dyn SKLazySource<F, Error = SKError> + Send + 'pass>,
}

impl<F: SKFloat> SKLazySource<F> for SKBoxedSource<'_, F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        self.inner.next_batch()
    }
}

/// Every provider whose error converts into the taxonomy is an object provider.
impl<F, Provider> SKStreamingProvider<F> for Provider
where
    F: SKFloat,
    Provider: SKSourceProvider<F>,
{
    fn open_object_source(&mut self) -> Result<SKBoxedSource<'_, F>, SKError> {
        let source = self.open()?;
        Ok(SKBoxedSource {
            inner: Box::new(SKErrorMappingSource { source }),
        })
    }
}

/// A source wrapper that maps its native error into the central taxonomy.
struct SKErrorMappingSource<Source> {
    source: Source,
}

impl<F, Source> SKLazySource<F> for SKErrorMappingSource<Source>
where
    Source: SKLazySource<F> + Send,
    Source::Error: Into<SKError>,
{
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        self.source.next_batch().map_err(Into::into)
    }
}

/// A concrete [`SKSourceProvider`] over an object-safe provider.
///
/// Its `IS_REOPENABLE` is `true` by construction: the pipeline only reaches
/// the stage adapter after materializing a one-shot source (or with a
/// reopenable provider in hand), so a stage may safely ask the shared executor
/// for multiple passes.
pub(super) struct SKObjectProvider<'pass, F: SKFloat> {
    pub(super) inner: &'pass mut dyn SKStreamingProvider<F>,
}

impl<F: SKFloat> SKSourceProvider<F> for SKObjectProvider<'_, F> {
    type Error = SKError;

    type Source<'pass>
        = SKBoxedSource<'pass, F>
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        self.inner.open_object_source()
    }
}

/// A provider that transforms every upstream batch through the fitted stages.
///
/// Stage `i` is fitted through this wrapper holding stages `0..i`, so it sees
/// exactly the blocks the previous stages produce.
pub(super) struct SKTransformingProvider<'stages, F: SKFloat, Provider> {
    pub(super) upstream: &'stages mut Provider,
    pub(super) stages: &'stages [Box<dyn SKFittedStage<F>>],
}

impl<F, Provider> SKSourceProvider<F> for SKTransformingProvider<'_, F, Provider>
where
    F: SKFloat,
    Provider: SKSourceProvider<F>,
{
    type Error = SKError;

    type Source<'pass>
        = SKTransformingSource<'pass, F, Provider::Source<'pass>>
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = Provider::IS_REOPENABLE;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        let upstream = self.upstream.open()?;
        Ok(SKTransformingSource {
            upstream,
            stages: self.stages,
        })
    }
}

/// One pass of a [`SKTransformingProvider`].
pub(super) struct SKTransformingSource<'stages, F: SKFloat, Source> {
    upstream: Source,
    stages: &'stages [Box<dyn SKFittedStage<F>>],
}

impl<F, Source> SKLazySource<F> for SKTransformingSource<'_, F, Source>
where
    F: SKFloat,
    Source: SKLazySource<F> + Send,
    Source::Error: Into<SKError>,
{
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        let Some(batch) = self.upstream.next_batch().map_err(Into::into)? else {
            return Ok(None);
        };
        let position = batch.position();
        let is_final = batch.is_final();
        let mut block = SKFeatureBlock::from(batch.view());
        for (index, stage) in self.stages.iter().enumerate() {
            block = stage
                .transform_stage_batch(&block)
                .map_err(|error| super::errors::sk_stage_failure(index, error))?;
        }
        Ok(Some(super::fitting::sk_data_batch_from_block(
            block, position, is_final,
        )?))
    }
}
