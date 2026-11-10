//! The pipeline builder: the only construction surface (spec `typed-pipeline`).

use std::marker::PhantomData;

use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKSupervisedFit, SKFloat};

use super::{SKPipelineClassifierBuilder, SKPipelineRegressorBuilder};
use crate::{SKPipelineTransformer, stages::SKConfiguredStages};

/// The entry point of every typed pipeline.
///
/// [`SKPipeline::start`] opens a builder whose stage tuple is empty; each
/// [`SKPipelineBuilder::then`] appends a stage to the tuple, so the type records
/// the composition and an incompatible stage fails to compile.
pub struct SKPipeline;

impl SKPipeline {
    /// Start a pipeline in the scalar `F`.
    pub fn start<F: SKFloat>() -> SKPipelineBuilder<F, ()> {
        SKPipelineBuilder {
            stages: (),
            backend: SKBackendKind::Faer,
            execution_intent: SKExecutionMode::Automatic,
            marker: PhantomData,
        }
    }
}

/// A pipeline under construction: a typed tuple of configured stages plus the
/// backend and execution intent every operation records.
pub struct SKPipelineBuilder<F: SKFloat, Stages> {
    pub(super) stages: Stages,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
    pub(super) marker: PhantomData<F>,
}

impl<F: SKFloat, Stages> SKPipelineBuilder<F, Stages> {
    /// Append a transformer stage, extending the recorded tuple.
    ///
    /// The stage must implement [`SKPipelineTransformer`] in the pipeline
    /// scalar `F`; a stage that only exists for another scalar fails here, at
    /// compile time.
    pub fn then<Stage>(self, stage: Stage) -> SKPipelineBuilder<F, (Stages, Stage)>
    where
        Stage: SKPipelineTransformer<F>,
    {
        SKPipelineBuilder {
            stages: (self.stages, stage),
            backend: self.backend,
            execution_intent: self.execution_intent,
            marker: PhantomData,
        }
    }

    /// The backend kind recorded in every pipeline operation span (default
    /// `Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The execution intent recorded in every pipeline operation span (default
    /// `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        if let Ok(mode) = mode.try_into() {
            self.execution_intent = mode;
        }
        self
    }
}

impl<F, Stages> SKPipelineBuilder<F, Stages>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    /// Finish the chain with a supervised classifier.
    ///
    /// The final estimator's fitted model must implement the shared classifier
    /// prediction contract, so a regressor used as a classifier finalizer fails
    /// to compile. An empty stage list is rejected with
    /// [`SKError::InvalidHyperparameter`].
    pub fn finalize_classifier<Estimator>(
        self,
        estimator: Estimator,
    ) -> Result<SKPipelineClassifierBuilder<F, Stages, Estimator>, SKError>
    where
        Estimator: SKSupervisedFit<F>,
        Estimator::Error: Into<SKError>,
        Estimator::Model: sciencekit_common::SKClassifierPredictor<F>,
        <Estimator::Model as sciencekit_common::SKClassifierPredictor<F>>::Error: Into<SKError>,
    {
        self.validate_number_of_stages()?;
        Ok(SKPipelineClassifierBuilder {
            stages: self.stages,
            estimator,
            backend: self.backend,
            execution_intent: self.execution_intent,
            marker: PhantomData,
        })
    }

    /// Finish the chain with a supervised regressor.
    ///
    /// The final estimator's fitted model must implement the shared regressor
    /// prediction contract. An empty stage list is rejected with
    /// [`SKError::InvalidHyperparameter`].
    pub fn finalize_regressor<Estimator>(
        self,
        estimator: Estimator,
    ) -> Result<SKPipelineRegressorBuilder<F, Stages, Estimator>, SKError>
    where
        Estimator: SKSupervisedFit<F>,
        Estimator::Error: Into<SKError>,
        Estimator::Model: sciencekit_common::SKRegressorPredictor<F>,
        <Estimator::Model as sciencekit_common::SKRegressorPredictor<F>>::Error: Into<SKError>,
    {
        self.validate_number_of_stages()?;
        Ok(SKPipelineRegressorBuilder {
            stages: self.stages,
            estimator,
            backend: self.backend,
            execution_intent: self.execution_intent,
            marker: PhantomData,
        })
    }

    fn validate_number_of_stages(&self) -> Result<(), SKError> {
        if self.stages.number_of_stages() == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "stages",
                reason: "a pipeline requires at least one transformer stage".to_string(),
            });
        }
        Ok(())
    }
}
