//! The configured and fitted streaming pipelines (spec `streaming-pipeline`).
//!
//! A configured pipeline owns the erased stage list, the batch finalizer
//! estimator and the execution intent. Fitting walks the stages in order —
//! each one reading the previous stage's transformed batches — then performs a
//! final bounded pass that accumulates the transformed blocks for the
//! estimator. The fitted pipeline keeps the stage models and transforms one
//! batch at a time.

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat, SKIntoFeatureBlock,
    SKLazySource, SKOperationAttributes, SKOperationKind, SKSourceProvider, SKUnsupervisedFit,
    observability::sk_run_operation,
};

use crate::SKFeatureBlock;

use super::errors::sk_reject_one_shot_multiple_epochs;
use super::finalizer::SKFinalizerBuffer;
use super::fitting::{sk_apply_fitted_stages, sk_materialize};
use super::provider::SKTransformingProvider;
use super::stage::{SKFittedStage, SKStreamingStage};

/// A streaming pipeline under construction with its chosen finalizer.
pub struct SKStreamingPipelineConfigured<F: SKFloat, Estimator> {
    pub(super) stages: Vec<Box<dyn SKStreamingStage<F>>>,
    pub(super) estimator: Estimator,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
    pub(super) epochs: usize,
    pub(super) memory_budget_bytes: usize,
}

impl<F, Estimator> SKStreamingPipelineConfigured<F, Estimator>
where
    F: SKFloat,
    Estimator: SKUnsupervisedFit<F>,
    Estimator::Error: Into<SKError>,
{
    /// Fit every stage and then the batch finalizer.
    ///
    /// A reopenable provider is read once per stage plus once for the
    /// finalizer. A one-shot provider is read exactly once into memory — within
    /// the configured budget — and then served from the materialized blocks; a
    /// one-shot provider with more than one epoch is rejected because it cannot
    /// replay its source.
    pub fn fit<Provider>(
        &self,
        provider: &mut Provider,
    ) -> Result<SKFittedStreamingPipeline<F, Estimator::Model>, SKError>
    where
        Provider: SKSourceProvider<F>,
    {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: 0,
            columns_count: 0,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            if Provider::IS_REOPENABLE {
                return self.sk_fit_reopenable(provider, 0);
            }
            if self.epochs > 1 {
                return Err(sk_reject_one_shot_multiple_epochs(self.epochs));
            }
            let (mut materialized, materialized_bytes) =
                sk_materialize(provider, self.memory_budget_bytes)?;
            self.sk_fit_reopenable(&mut materialized, materialized_bytes)
        })
    }

    fn sk_fit_reopenable<Provider>(
        &self,
        provider: &mut Provider,
        materialized_bytes: usize,
    ) -> Result<SKFittedStreamingPipeline<F, Estimator::Model>, SKError>
    where
        Provider: SKSourceProvider<F>,
    {
        let mut fitted_stages: Vec<Box<dyn SKFittedStage<F>>> =
            Vec::with_capacity(self.stages.len());
        for index in 0..self.stages.len() {
            let model = {
                let mut transformed = SKTransformingProvider {
                    upstream: provider,
                    stages: &fitted_stages,
                };
                self.stages[index].fit_stage(&mut transformed, self.epochs)?
            };
            fitted_stages.push(model);
        }
        let mut buffer = SKFinalizerBuffer::new(self.memory_budget_bytes);
        let mut source = provider.open()?;
        let mut number_of_batches = 0;
        let mut number_of_rows = 0;
        while let Some(batch) = source.next_batch().map_err(Into::into)? {
            number_of_batches += 1;
            number_of_rows += batch.number_of_rows();
            let transformed =
                sk_apply_fitted_stages(&fitted_stages, SKFeatureBlock::from(batch.view()))?;
            buffer.push(transformed)?;
        }
        let number_of_features = buffer.columns().unwrap_or(0);
        let finalizer_bytes = buffer.used_bytes();
        let accumulated = buffer.finish()?;
        let final_model = self.estimator.fit(&accumulated).map_err(Into::into)?;
        Ok(SKFittedStreamingPipeline {
            stages: fitted_stages,
            final_model,
            number_of_features,
            backend: self.backend,
            execution_intent: self.execution_intent,
            report: SKStreamingFitReport {
                number_of_stages: self.stages.len(),
                number_of_epochs: self.epochs,
                number_of_batches,
                number_of_rows,
                materialized_bytes,
                finalizer_bytes,
            },
        })
    }
}

/// A fitted streaming pipeline.
pub struct SKFittedStreamingPipeline<F: SKFloat, FinalModel> {
    pub(super) stages: Vec<Box<dyn SKFittedStage<F>>>,
    pub(super) final_model: FinalModel,
    pub(super) number_of_features: usize,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
    pub(super) report: SKStreamingFitReport,
}

impl<F: SKFloat, FinalModel> SKFittedStreamingPipeline<F, FinalModel> {
    /// The number of transformer stages in the chain.
    pub fn number_of_stages(&self) -> usize {
        self.stages.len()
    }

    /// The number of features the finalizer was fitted on.
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// The backend kind recorded in every operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent recorded in every operation span.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_intent
    }

    /// The fit report describing the streaming run.
    pub fn report(&self) -> SKStreamingFitReport {
        self.report
    }

    /// The fitted batch-finalizer model.
    pub fn final_model(&self) -> &FinalModel {
        &self.final_model
    }
}

impl<F, FinalModel> SKFittedStreamingPipeline<F, FinalModel>
where
    F: SKFloat,
    FinalModel: SKFeatureTransformer<F>,
    FinalModel::Error: Into<SKError>,
    FinalModel::Output: SKIntoFeatureBlock<F>,
{
    /// Transform one batch through every fitted stage, then the finalizer.
    ///
    /// The input is borrowed and copied once into the owned block medium; the
    /// chain then moves blocks without an intermediate copy and never densifies
    /// a sparse batch it receives.
    pub fn transform_batch(&self, batch: &SKFeatureBlock<F>) -> Result<SKFeatureBlock<F>, SKError> {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count: batch.number_of_rows(),
            columns_count: batch.number_of_columns(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let transformed =
                sk_apply_fitted_stages(&self.stages, SKFeatureBlock::from(batch.view()))?;
            self.final_model
                .transform(&transformed)
                .map(SKIntoFeatureBlock::into_feature_block)
                .map_err(Into::into)
        })
    }
}

/// Metrics describing one streaming fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKStreamingFitReport {
    pub(super) number_of_stages: usize,
    pub(super) number_of_epochs: usize,
    pub(super) number_of_batches: usize,
    pub(super) number_of_rows: usize,
    pub(super) materialized_bytes: usize,
    pub(super) finalizer_bytes: usize,
}

impl SKStreamingFitReport {
    /// The number of transformer stages fitted.
    pub fn number_of_stages(&self) -> usize {
        self.number_of_stages
    }

    /// The configured epoch count (one for a one-shot provider).
    pub fn number_of_epochs(&self) -> usize {
        self.number_of_epochs
    }

    /// The number of batches read in the finalizer pass.
    pub fn number_of_batches(&self) -> usize {
        self.number_of_batches
    }

    /// The number of rows read in the finalizer pass.
    pub fn number_of_rows(&self) -> usize {
        self.number_of_rows
    }

    /// The resident bytes of a materialized one-shot source, zero when the
    /// provider was reopened directly.
    pub fn materialized_bytes(&self) -> usize {
        self.materialized_bytes
    }

    /// The resident bytes the finalizer accumulated.
    pub fn finalizer_bytes(&self) -> usize {
        self.finalizer_bytes
    }
}
