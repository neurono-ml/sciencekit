//! The streaming pipeline builder: the only construction surface.
//!
//! Stages are appended in order and erased behind object-safe adapters; the
//! batch finalizer is chosen last. Validation (at least one stage, a non-zero
//! epoch count, a non-zero memory budget) happens when the finalizer is
//! attached, so an invalid pipeline never reaches `fit`.

use std::sync::Arc;

use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat};

use super::errors::{sk_validate_epochs, sk_validate_memory_budget};
use super::pipeline::SKStreamingPipelineConfigured;
use super::stage::SKStreamingStage;
use super::transformer::SKStreamingTransformer;

/// The default resident-byte budget for materialized intermediates (64 MiB).
const DEFAULT_MEMORY_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// The entry point of every streaming pipeline.
///
/// [`SKStreamingPipeline::start`] opens a builder in the scalar `F`; each
/// [`SKStreamingPipelineBuilder::then`] appends a streaming transformer and
/// [`SKStreamingPipelineBuilder::finalize_estimator`] attaches the batch
/// finalizer.
pub struct SKStreamingPipeline;

impl SKStreamingPipeline {
    /// Start a streaming pipeline in the scalar `F`.
    pub fn start<F: SKFloat>() -> SKStreamingPipelineBuilder<F> {
        SKStreamingPipelineBuilder {
            stages: Vec::new(),
            backend: SKBackendKind::Faer,
            execution_intent: SKExecutionMode::Automatic,
            epochs: 1,
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

/// A streaming pipeline under construction.
pub struct SKStreamingPipelineBuilder<F: SKFloat> {
    pub(super) stages: Vec<Box<dyn SKStreamingStage<F>>>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
    pub(super) epochs: usize,
    pub(super) memory_budget_bytes: usize,
}

impl<F: SKFloat> SKStreamingPipelineBuilder<F> {
    /// Append a streaming transformer stage.
    ///
    /// The stage must be safely shareable and own its fitted model, because the
    /// fitted pipeline is transformed concurrently. The configured transformer
    /// is kept inside the fitted stage, so `transform_batch` can run it later.
    pub fn then<Transformer>(mut self, transformer: Transformer) -> Self
    where
        Transformer: SKStreamingTransformer<F> + Send + Sync + 'static,
        Transformer::Model: Send + Sync + 'static,
        Transformer::Error: Into<SKError>,
    {
        self.stages.push(Box::new(Arc::new(transformer)));
        self
    }

    /// The backend kind recorded in every pipeline operation span (default
    /// [`SKBackendKind::Faer`]).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The execution intent recorded in every pipeline operation span (default
    /// [`SKExecutionMode::Automatic`]).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        if let Ok(mode) = mode.try_into() {
            self.execution_intent = mode;
        }
        self
    }

    /// The number of passes each stage reads from a reopenable provider
    /// (default `1`; must be at least `1`).
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// The resident-byte budget for materialized intermediates (default 64
    /// MiB; must be non-zero).
    pub fn memory_budget_bytes(mut self, memory_budget_bytes: usize) -> Self {
        self.memory_budget_bytes = memory_budget_bytes;
        self
    }

    /// Attach the batch finalizer and validate the accumulated configuration.
    ///
    /// An empty stage list, zero epochs and a zero memory budget are rejected
    /// with [`SKError::InvalidHyperparameter`].
    pub fn finalize_estimator<Estimator>(
        self,
        estimator: Estimator,
    ) -> Result<SKStreamingPipelineConfigured<F, Estimator>, SKError> {
        if self.stages.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "stages",
                reason: "a streaming pipeline requires at least one transformer stage".to_string(),
            });
        }
        sk_validate_epochs(self.epochs)?;
        sk_validate_memory_budget(self.memory_budget_bytes)?;
        Ok(SKStreamingPipelineConfigured {
            stages: self.stages,
            estimator,
            backend: self.backend,
            execution_intent: self.execution_intent,
            epochs: self.epochs,
            memory_budget_bytes: self.memory_budget_bytes,
        })
    }
}
