//! Regressor finalization: the builder that owns a regressor final estimator
//! and the `fit` that produces the fitted regressor pipeline.

use std::marker::PhantomData;

use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFeatureBlock, SKFloat, SKOperationKind,
    SKRegressorPredictor, SKSupervisedFit, SKTargetView, sk_run_operation,
};

use crate::stages::SKConfiguredStages;

use super::fit::sk_fit_final;
use super::model::SKPipelineRegressor;
use super::predict::sk_pipeline_attributes;

/// A pipeline whose final stage is a supervised regressor.
///
/// [`SKPipelineRegressorBuilder::fit`] walks the transformer stages forward and
/// fits the final estimator on the last block.
pub struct SKPipelineRegressorBuilder<F: SKFloat, Stages, Estimator> {
    pub(super) stages: Stages,
    pub(super) estimator: Estimator,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
    pub(super) marker: PhantomData<F>,
}

impl<F, Stages, Estimator> SKPipelineRegressorBuilder<F, Stages, Estimator>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
    Estimator: SKSupervisedFit<F>,
    Estimator::Error: Into<SKError>,
    Estimator::Model: SKRegressorPredictor<F>,
    <Estimator::Model as SKRegressorPredictor<F>>::Error: Into<SKError>,
{
    /// Fit every transformer stage on the current training data, transform the
    /// data forward and fit the final regressor on the last block.
    pub fn fit<'a, X, T>(
        self,
        features: X,
        targets: T,
    ) -> Result<SKPipelineRegressor<F, Stages, Estimator::Model>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let targets = targets.try_into()?;
        let backend = self.backend;
        let execution_intent = self.execution_intent;
        let attributes =
            sk_pipeline_attributes(SKOperationKind::Fit, &view, execution_intent, backend);
        let stages = self.stages;
        let estimator = self.estimator;
        sk_run_operation(attributes, move || {
            let block = SKFeatureBlock::from(view);
            let (fitted_stages, last_block) = stages.fit_stages(&block)?;
            let final_model = sk_fit_final(&estimator, &last_block, targets)?;
            let number_of_features = last_block.number_of_columns();
            Ok(SKPipelineRegressor::new(
                stages,
                fitted_stages,
                final_model,
                number_of_features,
                backend,
                execution_intent,
            ))
        })
    }
}
