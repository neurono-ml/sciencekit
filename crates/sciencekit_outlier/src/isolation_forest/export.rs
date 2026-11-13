//! Format-agnostic state export/import for [`SKIsolationForestModel`].

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, sk_backend_parameter, sk_float_parameter,
    sk_integer_parameter, sk_model_tensor_data_type, sk_parse_backend_parameter,
    sk_validate_import_metadata,
};
use sciencekit_tree::export_helpers::{export_tree, import_tree};

use super::model::SKIsolationForestModel;

/// Stable estimator identifier for exported isolation-forest states.
pub const SK_ISOLATION_FOREST_ESTIMATOR_NAME: &str = "SKIsolationForest";

impl<F: SKFloat> SKExportable<F> for SKIsolationForestModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_ISOLATION_FOREST_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.feature_count,
        )?
        .with_parameter(
            "number_of_trees",
            SKParameterValue::from(self.trees.len() as i64),
        )?
        .with_parameter(
            "subsample_size",
            SKParameterValue::from(self.subsample_size as i64),
        )?
        .with_parameter(
            "offset",
            SKParameterValue::from(self.offset.to_f64().unwrap_or(-0.5)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut tensors = Vec::new();
        for (index, tree) in self.trees.iter().enumerate() {
            tensors.extend(export_tree(
                tree,
                &format!("isolation_forest.tree.{index}"),
            )?);
        }
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKIsolationForestModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_ISOLATION_FOREST_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let number_of_trees = sk_integer_parameter(metadata, "number_of_trees", 0).max(0) as usize;
        let mut trees = Vec::with_capacity(number_of_trees);
        for index in 0..number_of_trees {
            trees.push(import_tree(
                state,
                &format!("isolation_forest.tree.{index}"),
                feature_count,
            )?);
        }
        Ok(SKIsolationForestModel {
            trees,
            subsample_size: sk_integer_parameter(metadata, "subsample_size", 1).max(1) as usize,
            feature_count,
            offset: F::from(sk_float_parameter(metadata, "offset", -0.5))
                .ok_or(SKError::Conversion("offset is not finite".to_string()))?,
            backend: sk_parse_backend_parameter(metadata),
            execution_mode: SKExecutionMode::Automatic,
        })
    }
}
