//! The streaming transformer contract (spec `streaming-pipeline`, Decision 1).
//!
//! A streaming transformer learns its state by consuming a source through the
//! shared prefetch executor and applies that state to one owned batch at a
//! time. It never retains the dataset: the returned model owns only the learned
//! state, and [`SKStreamingTransformer::transform_batch`] borrows it.

use sciencekit_common::{SKError, SKFeatureBlock, SKFloat, SKSourceProvider};

/// A transformer that fits from a streaming source and transforms owned
/// batches.
///
/// The two halves of the contract mirror the batch pipeline's fit/transform
/// split: [`fit_streaming`](Self::fit_streaming) drives the shared executor and
/// returns a model bearing only the learned state,
/// [`transform_batch`](Self::transform_batch) applies that model to one
/// [`SKFeatureBlock`]. The block's representation is the stage's
/// responsibility: a stage that accepts a CSR batch must return CSR rather than
/// densifying it.
pub trait SKStreamingTransformer<F: SKFloat> {
    /// The fitted model: the sole bearer of the learned state.
    type Model;
    /// The operation error type; converts from the central [`SKError`].
    type Error: From<SKError>;

    /// Fit the transformer by reading `source` for `epochs` passes.
    ///
    /// A reopenable provider opens a fresh source per pass; a one-shot provider
    /// yields a single pass. The implementation drives
    /// [`sk_run_streaming_provider`](sciencekit_common::execution::sk_run_streaming_provider)
    /// with an
    /// [`SKStreamingPassPolicy::Iterative`](sciencekit_common::execution::SKStreamingPassPolicy)
    /// policy so the epoch budget is enforced by the shared executor.
    fn fit_streaming<S: SKSourceProvider<F>>(
        &self,
        source: &mut S,
        epochs: usize,
    ) -> Result<Self::Model, Self::Error>;

    /// Transform one owned batch with the fitted model.
    ///
    /// The returned block has the same row count as the input and keeps the
    /// representation the stage produces.
    fn transform_batch(
        &self,
        model: &Self::Model,
        batch: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, Self::Error>;
}
