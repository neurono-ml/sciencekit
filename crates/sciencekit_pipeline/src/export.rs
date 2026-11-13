//! Format-agnostic export of the typed pipelines.
//!
//! A fitted typed pipeline owns concrete stage models and a concrete final
//! model, so their states nest under `pipeline.stage.<index>` and
//! `pipeline.final`. The configured stage chain cannot be rebuilt from a state
//! — it holds live estimator values, not data — so import is deliberately not
//! implemented for the typed pipeline; the graph pipeline exposes the same
//! limitation through its type-erased boxes.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExportable, SKFloat, SKImportable, SKModelMetadata,
    SKModelState, SKParameterValue, sk_backend_parameter, sk_embed_state,
    sk_model_tensor_data_type, sk_validate_import_metadata,
};

use crate::pipeline::{SKPipelineClassifier, SKPipelineRegressor};
use crate::stages::SKConfiguredStages;

/// Stable estimator identifier for exported classifier-pipeline states.
pub const SK_PIPELINE_CLASSIFIER_ESTIMATOR_NAME: &str = "SKPipelineClassifier";
/// Stable estimator identifier for exported regressor-pipeline states.
pub const SK_PIPELINE_REGRESSOR_ESTIMATOR_NAME: &str = "SKPipelineRegressor";

/// A fitted stage tuple that can describe every stage as a model state.
pub trait SKExportableStages<F: SKFloat> {
    /// Export every fitted stage in composition order.
    ///
    /// # Errors
    ///
    /// Returns the first stage export failure.
    fn export_stage_states(&self) -> Result<Vec<SKModelState<F>>, SKError>;
}

impl<F: SKFloat> SKExportableStages<F> for () {
    fn export_stage_states(&self) -> Result<Vec<SKModelState<F>>, SKError> {
        Ok(Vec::new())
    }
}

impl<F, Rest, Last> SKExportableStages<F> for (Rest, Last)
where
    F: SKFloat,
    Rest: SKExportableStages<F>,
    Last: SKExportable<F>,
{
    fn export_stage_states(&self) -> Result<Vec<SKModelState<F>>, SKError> {
        let mut states = self.0.export_stage_states()?;
        states.push(self.1.export_state()?);
        Ok(states)
    }
}

impl<F, Stages, FinalModel> SKExportable<F> for SKPipelineClassifier<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
    Stages::FittedStages: SKExportableStages<F>,
    FinalModel: SKExportable<F>,
{
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        export_pipeline(
            self.fitted_stages(),
            self.final_model(),
            self.number_of_features(),
            self.backend(),
            SK_PIPELINE_CLASSIFIER_ESTIMATOR_NAME,
        )
    }
}

impl<F, Stages, FinalModel> SKExportable<F> for SKPipelineRegressor<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
    Stages::FittedStages: SKExportableStages<F>,
    FinalModel: SKExportable<F>,
{
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        export_pipeline(
            self.fitted_stages(),
            self.final_model(),
            self.number_of_features(),
            self.backend(),
            SK_PIPELINE_REGRESSOR_ESTIMATOR_NAME,
        )
    }
}

/// Assemble the nested pipeline state.
fn export_pipeline<F, Stages, FinalModel>(
    fitted_stages: &Stages,
    final_model: &FinalModel,
    number_of_features: usize,
    backend: sciencekit_common::SKBackendKind,
    estimator_name: &str,
) -> Result<SKModelState<F>, SKError>
where
    F: SKFloat,
    Stages: SKExportableStages<F>,
    FinalModel: SKExportable<F>,
{
    let stage_states = fitted_stages.export_stage_states()?;
    let mut metadata = SKModelMetadata::new(
        estimator_name,
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        number_of_features,
    )?
    .with_parameter(
        "number_of_stages",
        SKParameterValue::from(stage_states.len() as i64),
    )?
    .with_parameter("backend", sk_backend_parameter(backend))?;
    let mut tensors = Vec::new();
    for (index, state) in stage_states.iter().enumerate() {
        metadata = sk_embed_state(
            metadata,
            &mut tensors,
            &format!("pipeline.stage.{index}"),
            state,
        )?;
    }
    let final_state = final_model.export_state()?;
    metadata = sk_embed_state(metadata, &mut tensors, "pipeline.final", &final_state)?;
    SKModelState::new(metadata, tensors)
}

/// Read the final stage count of an exported pipeline state.
///
/// # Errors
///
/// Returns the first [`SKError`] raised by validation.
pub fn sk_pipeline_stage_count<F: SKFloat>(
    state: &SKModelState<F>,
    estimator_name: &str,
) -> Result<i64, SKError> {
    sk_validate_import_metadata(
        state,
        estimator_name,
        SK_MODEL_FORMAT_VERSION,
        state.metadata().feature_count(),
    )?;
    Ok(sciencekit_common::sk_integer_parameter(
        state.metadata(),
        "number_of_stages",
        0,
    ))
}

impl<F, Stages, FinalModel> SKImportable<F> for SKPipelineClassifier<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    fn import_state(_state: &SKModelState<F>) -> Result<Self, SKError> {
        Err(sk_pipeline_import_unsupported())
    }
}

impl<F, Stages, FinalModel> SKImportable<F> for SKPipelineRegressor<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    fn import_state(_state: &SKModelState<F>) -> Result<Self, SKError> {
        Err(sk_pipeline_import_unsupported())
    }
}

fn sk_pipeline_import_unsupported() -> SKError {
    SKError::UnsupportedOperation {
        operation: "import_state",
        reason: "a typed pipeline state nests every fitted stage and the final model, but the                  configured stage chain holds live estimator values rather than data and cannot                  be rebuilt from the state; rebuild the pipeline in code and import each nested                  stage state"
            .to_string(),
    }
}
