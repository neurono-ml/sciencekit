//! Decision-tree and random-forest exporters through the ONNX-ML tree
//! ensembles (spec `onnx-export`, requirement "Per-family operators").
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod ensemble;
mod layout;

use sciencekit_common::{SKError, SKFloat, SKModelState, SKTensorDataType};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::helpers::{integer_parameter, string_parameter};
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};
use ensemble::SKOnnxTreeEnsemble;
use layout::SKOnnxTreeLayout;

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
        return Err(super::helpers::unsupported(
            estimator_name,
            "the ONNX-ML tree-ensemble operator stores thresholds and leaf values as float attributes, so only f32 trees are exported",
        ));
    }
    let output = match estimator_name {
        SK_DECISION_TREE_CLASSIFIER_NAME => {
            append_single_classifier(state, builder, input, namespace)?
        }
        SK_DECISION_TREE_REGRESSOR_NAME => {
            append_single_regressor(state, builder, input, namespace)?
        }
        SK_RANDOM_FOREST_CLASSIFIER_NAME => {
            append_forest_classifier(state, builder, input, namespace)?
        }
        _ => append_forest_regressor(state, builder, input, namespace)?,
    };
    Ok(Some(output))
}

fn append_single_classifier<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let classes = class_count(state);
    let layout = SKOnnxTreeLayout::read(state, "tree")?;
    let mut ensemble = SKOnnxTreeEnsemble::default();
    ensemble.append_tree(&layout, 0, 1.0, Some(classes), None);
    builder.add_node(ensemble.classifier_node(input, classes, namespace));
    add_label_from_probabilities(builder, namespace);
    Ok(classification_output(namespace, classes))
}

fn append_single_regressor<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let layout = SKOnnxTreeLayout::read(state, "tree")?;
    let mut ensemble = SKOnnxTreeEnsemble::default();
    ensemble.append_tree(&layout, 0, 1.0, None, None);
    add_regressor_node(builder, &ensemble, input, namespace)
}

fn append_forest_classifier<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let classes = class_count(state);
    let tree_count = integer_parameter(state, "number_of_trees", 0).max(1) as usize;
    let scale = 1.0 / tree_count as f64;
    let mut ensemble = SKOnnxTreeEnsemble::default();
    for index in 0..tree_count {
        let prefix = format!("forest.tree.{index}");
        if string_parameter(state, &format!("{prefix}.kind"), "fitted") == "pure_leaf" {
            let class =
                integer_parameter(state, &format!("{prefix}.class_index"), 0).max(0) as usize;
            ensemble.append_pure_leaf(index as i64, class, scale);
            continue;
        }
        let local_to_forest: Vec<usize> = state
            .metadata()
            .parameter(&format!("{prefix}.local_to_forest"))
            .and_then(sciencekit_common::SKParameterValue::as_list)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_integer())
                    .map(|value| value.max(0) as usize)
                    .collect()
            })
            .unwrap_or_default();
        let layout = SKOnnxTreeLayout::read(state, &format!("{prefix}.t.tree"))?;
        ensemble.append_tree(
            &layout,
            index as i64,
            scale,
            Some(classes),
            Some(&local_to_forest),
        );
    }
    builder.add_node(ensemble.classifier_node(input, classes, namespace));
    add_label_from_probabilities(builder, namespace);
    Ok(classification_output(namespace, classes))
}

fn append_forest_regressor<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let tree_count = integer_parameter(state, "number_of_trees", 0).max(1) as usize;
    let scale = 1.0 / tree_count as f64;
    let mut ensemble = SKOnnxTreeEnsemble::default();
    for index in 0..tree_count {
        let layout = SKOnnxTreeLayout::read(state, &format!("forest.tree.{index}.t.tree"))?;
        ensemble.append_tree(&layout, index as i64, scale, None, None);
    }
    add_regressor_node(builder, &ensemble, input, namespace)
}

fn add_regressor_node(
    builder: &mut SKOnnxGraphBuilder,
    ensemble: &SKOnnxTreeEnsemble,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let output = format!("{namespace}prediction");
    let node: SKOnnxNode = ensemble.regression_node(input, &output);
    builder.add_node(node);
    Ok(SKOnnxStageOutput {
        tensor: output,
        width: 1,
        kind: SKOnnxStageKind::Regression,
    })
}

fn add_label_from_probabilities(builder: &mut SKOnnxGraphBuilder, namespace: &str) {
    builder.add_node(
        SKOnnxNode::new(
            "ArgMax",
            vec![format!("{namespace}probabilities")],
            vec![format!("{namespace}label")],
        )
        .attribute_integer("axis", 1)
        .attribute_integer("keepdims", 0),
    );
}

fn classification_output(namespace: &str, classes: usize) -> SKOnnxStageOutput {
    SKOnnxStageOutput {
        tensor: format!("{namespace}probabilities"),
        width: classes,
        kind: SKOnnxStageKind::Classification {
            label_tensor: format!("{namespace}label"),
            probability_tensor: format!("{namespace}probabilities"),
            class_count: classes,
        },
    }
}

fn class_count<F: SKFloat>(state: &SKModelState<F>) -> usize {
    state
        .metadata()
        .class_labels()
        .map(<[String]>::len)
        .unwrap_or_else(|| integer_parameter(state, "number_of_classes", 2).max(1) as usize)
}
