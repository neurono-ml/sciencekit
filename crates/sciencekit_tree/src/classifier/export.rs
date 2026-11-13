//! Format-agnostic state export/import for [`SKDecisionTreeClassifierModel`].

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_canonicalize_labels, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::SKDecisionTreeClassifierModel;
use crate::export_helpers::{criterion_from_name, criterion_name, export_tree, import_tree};

/// Stable estimator identifier for exported decision-tree-classifier states.
pub const SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME: &str = "SKDecisionTreeClassifier";
/// The tensor carrying the normalized feature importances.
pub const SK_DECISION_TREE_IMPORTANCES_TENSOR: &str = "tree.feature_importances";

impl<F: SKFloat> SKExportable<F> for SKDecisionTreeClassifierModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.tree.feature_count(),
        )?
        .with_class_labels(self.classes.labels().map(str::to_string).collect())?
        .with_parameter(
            "number_of_classes",
            SKParameterValue::from(self.number_of_classes as i64),
        )?
        .with_parameter(
            "criterion",
            SKParameterValue::from(criterion_name(self.criterion)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut tensors = export_tree(&self.tree, "tree")?;
        tensors.push(SKTensor::floating(
            SK_DECISION_TREE_IMPORTANCES_TENSOR,
            vec![self.feature_importances.len()],
            self.feature_importances.clone(),
        )?);
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKDecisionTreeClassifierModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let tree = import_tree(state, "tree", feature_count)?;
        let (_, feature_importances) =
            sk_read_floating_tensor(state, SK_DECISION_TREE_IMPORTANCES_TENSOR)?;
        let labels: Vec<String> = metadata
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        Ok(SKDecisionTreeClassifierModel {
            tree,
            classes,
            number_of_classes: sk_integer_parameter(metadata, "number_of_classes", 1).max(1)
                as usize,
            feature_importances,
            criterion: criterion_from_name(sk_string_parameter(metadata, "criterion", "gini")),
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}
