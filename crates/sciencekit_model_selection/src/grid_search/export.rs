//! Format-agnostic export of the grid-search winners.
//!
//! A grid search does not reconstruct its candidate configuration from a state
//! (the candidates are arbitrary estimator values), so it exports the refitted
//! winner and its selection metadata and deliberately implements no import.

use sciencekit_common::{
    SKError, SKExportable, SKFloat, SKImportable, SKModelMetadata, SKModelState, SKParameterValue,
    SKSupervisedFit,
};

use super::classifier::SKGridSearchCVClassifierModel;
use super::regressor::SKGridSearchCVRegressorModel;

/// Stable prefix for the selection metadata a grid-search export adds.
pub const SK_GRID_SEARCH_PARAMETER_PREFIX: &str = "grid_search.";

impl<F, E> SKExportable<F> for SKGridSearchCVRegressorModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    E::Model: SKExportable<F>,
{
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let model = self
            .best_model
            .as_ref()
            .ok_or_else(|| SKError::UnsupportedOperation {
                operation: "export_state",
                reason: "the grid search was built with refit(false), so no best model is stored"
                    .to_owned(),
            })?;
        sk_augment_with_selection(
            model.export_state()?,
            self.best_candidate_index() as i64,
            self.best_score(),
            self.results().fold_scores().nrows(),
            true,
        )
    }
}

impl<F, E> SKExportable<F> for SKGridSearchCVClassifierModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    E::Model: SKExportable<F>,
{
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let model = self
            .best_model
            .as_ref()
            .ok_or_else(|| SKError::UnsupportedOperation {
                operation: "export_state",
                reason: "the grid search was built with refit(false), so no best model is stored"
                    .to_owned(),
            })?;
        sk_augment_with_selection(
            model.export_state()?,
            self.best_candidate_index() as i64,
            self.best_score(),
            self.results().fold_scores().nrows(),
            true,
        )
    }
}

/// Copy a state and add the selection metadata without touching its tensors.
fn sk_augment_with_selection<F: SKFloat>(
    state: SKModelState<F>,
    best_candidate_index: i64,
    best_score: F,
    candidate_count: usize,
    refitted: bool,
) -> Result<SKModelState<F>, SKError> {
    let original = state.metadata();
    let mut metadata = SKModelMetadata::new(
        original.estimator_name().to_string(),
        original.format_version(),
        original.scalar_data_type(),
        original.feature_count(),
    )?;
    if let Some(labels) = original.class_labels() {
        metadata = metadata.with_class_labels(labels.to_vec())?;
    }
    for (name, value) in original.parameters() {
        metadata = metadata.with_parameter(name.clone(), value.clone())?;
    }
    metadata = metadata
        .with_parameter(
            format!("{SK_GRID_SEARCH_PARAMETER_PREFIX}best_candidate_index"),
            SKParameterValue::from(best_candidate_index),
        )?
        .with_parameter(
            format!("{SK_GRID_SEARCH_PARAMETER_PREFIX}best_score"),
            SKParameterValue::from(best_score.to_f64().unwrap_or(0.0)),
        )?
        .with_parameter(
            format!("{SK_GRID_SEARCH_PARAMETER_PREFIX}candidate_count"),
            SKParameterValue::from(candidate_count as i64),
        )?
        .with_parameter(
            format!("{SK_GRID_SEARCH_PARAMETER_PREFIX}refitted"),
            SKParameterValue::from(refitted),
        )?;
    SKModelState::new(metadata, state.tensors().values().cloned().collect())
}

impl<F, E> SKImportable<F> for SKGridSearchCVRegressorModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
{
    fn import_state(_state: &SKModelState<F>) -> Result<Self, SKError> {
        Err(sk_grid_search_import_unsupported())
    }
}

impl<F, E> SKImportable<F> for SKGridSearchCVClassifierModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
{
    fn import_state(_state: &SKModelState<F>) -> Result<Self, SKError> {
        Err(sk_grid_search_import_unsupported())
    }
}

fn sk_grid_search_import_unsupported() -> SKError {
    SKError::UnsupportedOperation {
        operation: "import_state",
        reason: "a grid-search state carries the refitted winner's tensors and the selection                  metadata but not the winner's own metadata, and the candidate configuration                  cannot be rebuilt from data; re-fit the search instead"
            .to_string(),
    }
}
