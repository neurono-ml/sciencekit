//! The fitted pipeline models: configured stages, their fitted twins, the
//! final model and the recorded operation metadata.

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::stages::SKConfiguredStages;

/// A fitted classifier pipeline.
///
/// It owns the configured stages (their settings are needed to transform new
/// data), the fitted model of every stage and the final classifier model. The
/// type parameters are the pipeline scalar, the configured stage tuple and the
/// final model.
pub struct SKPipelineClassifier<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    pub(super) stages: Stages,
    pub(super) fitted_stages: Stages::FittedStages,
    pub(super) final_model: FinalModel,
    pub(super) number_of_features: usize,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F, Stages, FinalModel> SKPipelineClassifier<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    pub(super) fn new(
        stages: Stages,
        fitted_stages: Stages::FittedStages,
        final_model: FinalModel,
        number_of_features: usize,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        Self {
            stages,
            fitted_stages,
            final_model,
            number_of_features,
            backend,
            execution_intent,
        }
    }

    /// The configured transformer stages, in composition order.
    pub fn stages(&self) -> &Stages {
        &self.stages
    }

    /// The fitted model of every transformer stage, in composition order.
    pub fn fitted_stages(&self) -> &Stages::FittedStages {
        &self.fitted_stages
    }

    /// The fitted final classifier model.
    pub fn final_model(&self) -> &FinalModel {
        &self.final_model
    }

    /// The number of transformer stages in the chain.
    pub fn number_of_stages(&self) -> usize {
        self.stages.number_of_stages()
    }

    /// The number of features the final estimator was fitted on.
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
}

/// A fitted regressor pipeline.
///
/// It owns the configured stages, their fitted models and the final regressor
/// model.
pub struct SKPipelineRegressor<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    pub(super) stages: Stages,
    pub(super) fitted_stages: Stages::FittedStages,
    pub(super) final_model: FinalModel,
    pub(super) number_of_features: usize,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F, Stages, FinalModel> SKPipelineRegressor<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    pub(super) fn new(
        stages: Stages,
        fitted_stages: Stages::FittedStages,
        final_model: FinalModel,
        number_of_features: usize,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
    ) -> Self {
        Self {
            stages,
            fitted_stages,
            final_model,
            number_of_features,
            backend,
            execution_intent,
        }
    }

    /// The configured transformer stages, in composition order.
    pub fn stages(&self) -> &Stages {
        &self.stages
    }

    /// The fitted model of every transformer stage, in composition order.
    pub fn fitted_stages(&self) -> &Stages::FittedStages {
        &self.fitted_stages
    }

    /// The fitted final regressor model.
    pub fn final_model(&self) -> &FinalModel {
        &self.final_model
    }

    /// The number of transformer stages in the chain.
    pub fn number_of_stages(&self) -> usize {
        self.stages.number_of_stages()
    }

    /// The number of features the final estimator was fitted on.
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
}
