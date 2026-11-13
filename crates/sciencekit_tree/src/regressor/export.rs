//! Format-agnostic state export/import for [`SKDecisionTreeRegressorModel`].

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_model_tensor_data_type, sk_parse_backend_parameter, sk_read_floating_tensor,
    sk_string_parameter, sk_validate_import_metadata,
};

use super::SKDecisionTreeRegressorModel;
use crate::export_helpers::{criterion_from_name, criterion_name, export_tree, import_tree};

/// Stable estimator identifier for exported decision-tree-regressor states.
pub const SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME: &str = "SKDecisionTreeRegressor";
/// The tensor carrying the normalized feature importances.
pub const SK_DECISION_TREE_REGRESSOR_IMPORTANCES_TENSOR: &str = "tree.feature_importances";

impl<F: SKFloat> SKExportable<F> for SKDecisionTreeRegressorModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.tree.feature_count(),
        )?
        .with_parameter(
            "criterion",
            SKParameterValue::from(criterion_name(self.criterion)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut tensors = export_tree(&self.tree, "tree")?;
        tensors.push(SKTensor::floating(
            SK_DECISION_TREE_REGRESSOR_IMPORTANCES_TENSOR,
            vec![self.feature_importances.len()],
            self.feature_importances.clone(),
        )?);
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKDecisionTreeRegressorModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let tree = import_tree(state, "tree", feature_count)?;
        let (_, feature_importances) =
            sk_read_floating_tensor(state, SK_DECISION_TREE_REGRESSOR_IMPORTANCES_TENSOR)?;
        let metadata = state.metadata();
        Ok(SKDecisionTreeRegressorModel {
            tree,
            feature_importances,
            criterion: criterion_from_name(sk_string_parameter(
                metadata,
                "criterion",
                "squared_error",
            )),
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}
