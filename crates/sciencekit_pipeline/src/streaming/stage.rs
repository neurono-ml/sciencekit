//! The object-safe stage adapter over [`SKStreamingTransformer`].
//!
//! The public transformer contract is generic over the provider type, so a
//! pipeline cannot store a heterogeneous `Vec` of transformers directly. This
//! module erases each configured stage into an object-safe
//! [`SKStreamingStage`] and each fitted model into an
//! [`SKFittedStage`], preserving the transformer next to its model so
//! [`SKStreamingTransformer::transform_batch`] can run later.

use std::sync::Arc;

use sciencekit_common::{SKError, SKFeatureBlock, SKFloat};

use super::provider::{SKObjectProvider, SKStreamingProvider};
use super::transformer::SKStreamingTransformer;

/// A configured stage erased behind a trait object.
pub(crate) trait SKStreamingStage<F: SKFloat>: Send + Sync {
    /// Fit the stage by reading the provider and return its boxed model.
    fn fit_stage(
        &self,
        provider: &mut dyn SKStreamingProvider<F>,
        epochs: usize,
    ) -> Result<Box<dyn SKFittedStage<F>>, SKError>;
}

/// A fitted stage that transforms one batch at a time.
pub(crate) trait SKFittedStage<F: SKFloat>: Send + Sync {
    /// Apply the fitted stage to one owned batch.
    fn transform_stage_batch(
        &self,
        batch: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError>;
}

/// Every shared streaming transformer is a stage when wrapped in an [`Arc`].
///
/// The `Arc` lets the fitted stage keep the configured transformer (needed by
/// [`SKStreamingTransformer::transform_batch`]) without requiring `Clone`.
impl<F, Transformer> SKStreamingStage<F> for Arc<Transformer>
where
    F: SKFloat,
    Transformer: SKStreamingTransformer<F> + Send + Sync + 'static,
    Transformer::Model: Send + Sync + 'static,
    Transformer::Error: Into<SKError>,
{
    fn fit_stage(
        &self,
        provider: &mut dyn SKStreamingProvider<F>,
        epochs: usize,
    ) -> Result<Box<dyn SKFittedStage<F>>, SKError> {
        let mut object_provider = SKObjectProvider { inner: provider };
        let model = self
            .fit_streaming(&mut object_provider, epochs)
            .map_err(Into::into)?;
        Ok(Box::new(SKBoxedFittedStage {
            transformer: Arc::clone(self),
            model,
        }))
    }
}

/// A fitted model paired with the configured transformer that produced it.
struct SKBoxedFittedStage<F, Transformer>
where
    F: SKFloat,
    Transformer: SKStreamingTransformer<F>,
{
    transformer: Arc<Transformer>,
    model: Transformer::Model,
}

impl<F, Transformer> SKFittedStage<F> for SKBoxedFittedStage<F, Transformer>
where
    F: SKFloat,
    Transformer: SKStreamingTransformer<F> + Send + Sync + 'static,
    Transformer::Model: Send + Sync + 'static,
    Transformer::Error: Into<SKError>,
{
    fn transform_stage_batch(
        &self,
        batch: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError> {
        self.transformer
            .transform_batch(&self.model, batch)
            .map_err(Into::into)
    }
}
