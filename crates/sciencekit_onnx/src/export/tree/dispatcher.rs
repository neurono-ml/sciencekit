//! The estimator-name dispatcher of the tree family.

use sciencekit_common::{SKError, SKFloat, SKModelState, SKTensorDataType};

use crate::graph::SKOnnxGraphBuilder;

use super::super::stage::SKOnnxStageOutput;
use super::{classifier, regressor};

pub(crate) const SK_DECISION_TREE_CLASSIFIER_NAME: &str = "SKDecisionTreeClassifier";
pub(crate) const SK_DECISION_TREE_REGRESSOR_NAME: &str = "SKDecisionTreeRegressor";
pub(crate) const SK_RANDOM_FOREST_CLASSIFIER_NAME: &str = "SKRandomForestClassifier";
pub(crate) const SK_RANDOM_FOREST_REGRESSOR_NAME: &str = "SKRandomForestRegressor";

/// Translate one tree-family estimator state, when the name matches.
pub(crate) fn append_tree<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    if !matches!(
        estimator_name,
        SK_DECISION_TREE_CLASSIFIER_NAME
            | SK_DECISION_TREE_REGRESSOR_NAME
            | SK_RANDOM_FOREST_CLASSIFIER_NAME
            | SK_RANDOM_FOREST_REGRESSOR_NAME
    ) {
        return Ok(None);
    }
    // ONNX-ML tree attributes are 32-bit floats; a double model would silently
    // lose threshold precision, so the exporter rejects it explicitly.
    if state.metadata().scalar_data_type() == SKTensorDataType::F64 {
        return Err(super::super::helpers::unsupported(
            estimator_name,
            "the ONNX-ML tree-ensemble operator stores thresholds and leaf values as float attributes, so only f32 trees are exported",
        ));
    }
    let output = match estimator_name {
        SK_DECISION_TREE_CLASSIFIER_NAME => {
            classifier::append_single_classifier(state, builder, input, namespace)?
        }
        SK_DECISION_TREE_REGRESSOR_NAME => {
            regressor::append_single_regressor(state, builder, input, namespace)?
        }
        SK_RANDOM_FOREST_CLASSIFIER_NAME => {
            classifier::append_forest_classifier(state, builder, input, namespace)?
        }
        _ => regressor::append_forest_regressor(state, builder, input, namespace)?,
    };
    Ok(Some(output))
}
