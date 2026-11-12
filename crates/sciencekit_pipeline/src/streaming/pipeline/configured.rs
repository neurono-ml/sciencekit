//! The configured pipeline: stages plus finalizer, before fitting.

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKLazySource, SKOperationAttributes,
    SKOperationKind, SKSourceProvider, SKUnsupervisedFit, observability::sk_run_operation,
};

use crate::SKFeatureBlock;

use super::super::errors::sk_reject_one_shot_multiple_epochs;
use super::super::finalizer::SKFinalizerBuffer;
use super::super::fitting::{sk_apply_fitted_stages, sk_materialize};
use super::super::provider::SKTransformingProvider;
use super::super::stage::{SKFittedStage, SKStreamingStage};
use super::fitted::SKFittedStreamingPipeline;
use super::report::SKStreamingFitReport;

/// A streaming pipeline under construction with its chosen finalizer.
pub struct SKStreamingPipelineConfigured<F: SKFloat, Estimator> {
    pub(crate) stages: Vec<Box<dyn SKStreamingStage<F>>>,
    pub(crate) estimator: Estimator,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_intent: SKExecutionMode,
    pub(crate) epochs: usize,
    pub(crate) memory_budget_bytes: usize,
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
        mut provider: Provider,
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
                return self.sk_fit_reopenable(&mut provider, 0);
            }
            if self.epochs > 1 {
                return Err(sk_reject_one_shot_multiple_epochs(self.epochs));
            }
            let (mut materialized, materialized_bytes) =
                sk_materialize(&mut provider, self.memory_budget_bytes)?;
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
