//! Classifier trees: leaf class weights accumulated per forest member.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::super::helpers::{integer_parameter, string_parameter};
use super::super::stage::{SKOnnxStageKind, SKOnnxStageOutput};
use super::ensemble::SKOnnxTreeEnsemble;
use super::layout::SKOnnxTreeLayout;

pub(crate) fn append_single_classifier<F: SKFloat>(
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

pub(crate) fn append_forest_classifier<F: SKFloat>(
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
