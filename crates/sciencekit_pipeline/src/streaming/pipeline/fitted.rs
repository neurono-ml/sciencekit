//! The fitted streaming pipeline and its per-batch transform.

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat, SKIntoFeatureBlock,
    SKOperationAttributes, SKOperationKind, observability::sk_run_operation,
};

use crate::SKFeatureBlock;

use super::super::fitting::sk_apply_fitted_stages;
use super::super::stage::SKFittedStage;
use super::report::SKStreamingFitReport;

/// A fitted streaming pipeline.
pub struct SKFittedStreamingPipeline<F: SKFloat, FinalModel> {
    pub(crate) stages: Vec<Box<dyn SKFittedStage<F>>>,
    pub(crate) final_model: FinalModel,
    pub(crate) number_of_features: usize,
    pub(crate) backend: SKBackendKind,
    pub(crate) execution_intent: SKExecutionMode,
    pub(crate) report: SKStreamingFitReport,
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
