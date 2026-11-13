//! Regressor trees: leaf responses accumulated per forest member.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::super::helpers::integer_parameter;
use super::super::stage::{SKOnnxStageKind, SKOnnxStageOutput};
use super::ensemble::SKOnnxTreeEnsemble;
use super::layout::SKOnnxTreeLayout;

pub(crate) fn append_single_regressor<F: SKFloat>(
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

pub(crate) fn append_forest_regressor<F: SKFloat>(
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
