//! The ONNX-ML tree-ensemble attributes accumulated across fitted trees.

use sciencekit_common::SKFloat;

use crate::graph::SKOnnxNode;

use super::layout::SKOnnxTreeLayout;

/// The attributes `TreeEnsembleClassifier`/`TreeEnsembleRegressor` expect.
#[derive(Default)]
pub(crate) struct SKOnnxTreeEnsemble {
    nodes_treeids: Vec<i64>,
    nodes_nodeids: Vec<i64>,
    nodes_featureids: Vec<i64>,
    nodes_modes: Vec<String>,
    nodes_values: Vec<f64>,
    nodes_truenodeids: Vec<i64>,
    nodes_falsenodeids: Vec<i64>,
    target_treeids: Vec<i64>,
    target_nodeids: Vec<i64>,
    target_weights: Vec<f64>,
    class_treeids: Vec<i64>,
    class_nodeids: Vec<i64>,
    class_ids: Vec<i64>,
    class_weights: Vec<f64>,
}

impl SKOnnxTreeEnsemble {
    /// Re-layout one tree under a tree id, scaling every leaf weight.
    pub(crate) fn append_tree<F: SKFloat>(
        &mut self,
        layout: &SKOnnxTreeLayout<F>,
        tree_id: i64,
        weight_scale: f64,
        classes: Option<usize>,
        local_to_forest: Option<&[usize]>,
    ) {
        for node in 0..layout.split_feature.len() {
            let is_leaf = layout.split_feature[node] < 0;
            self.push_node(layout, node, tree_id, is_leaf);
            if !is_leaf {
                continue;
            }
            match classes {
                Some(class_count) => {
                    for (local, value) in layout.leaf_values(node).iter().enumerate() {
                        let forest_class = local_to_forest
                            .and_then(|mapping| mapping.get(local).copied())
                            .unwrap_or(local);
                        if forest_class >= class_count {
                            continue;
                        }
                        self.class_treeids.push(tree_id);
                        self.class_nodeids.push(node as i64);
                        self.class_ids.push(forest_class as i64);
                        self.class_weights
                            .push(value.to_f64().unwrap_or(0.0) * weight_scale);
                    }
                }
                None => {
                    let value = layout
                        .leaf_values(node)
                        .first()
                        .copied()
                        .unwrap_or_else(F::zero);
                    self.target_treeids.push(tree_id);
                    self.target_nodeids.push(node as i64);
                    self.target_weights
                        .push(value.to_f64().unwrap_or(0.0) * weight_scale);
                }
            }
        }
    }

    /// Append a constant probability-1 leaf for a pure-leaf forest member.
    pub(crate) fn append_pure_leaf(&mut self, tree_id: i64, class: usize, weight_scale: f64) {
        self.nodes_treeids.push(tree_id);
        self.nodes_nodeids.push(0);
        self.nodes_featureids.push(0);
        self.nodes_modes.push("LEAF".to_string());
        self.nodes_values.push(0.0);
        self.nodes_truenodeids.push(0);
        self.nodes_falsenodeids.push(0);
        self.class_treeids.push(tree_id);
        self.class_nodeids.push(0);
        self.class_ids.push(class as i64);
        self.class_weights.push(weight_scale);
    }

    fn push_node<F: SKFloat>(
        &mut self,
        layout: &SKOnnxTreeLayout<F>,
        node: usize,
        tree_id: i64,
        is_leaf: bool,
    ) {
        self.nodes_treeids.push(tree_id);
        self.nodes_nodeids.push(node as i64);
        self.nodes_featureids.push(if is_leaf {
            0
        } else {
            layout.split_feature[node]
        });
        self.nodes_modes
            .push(if is_leaf { "LEAF" } else { "BRANCH_LEQ" }.to_string());
        self.nodes_values.push(if is_leaf {
            0.0
        } else {
            layout.threshold[node].to_f64().unwrap_or(0.0)
        });
        self.nodes_truenodeids
            .push(if is_leaf { 0 } else { layout.left[node] });
        self.nodes_falsenodeids
            .push(if is_leaf { 0 } else { layout.right[node] });
    }

    /// Build the classifier node over the accumulated attributes.
    pub(crate) fn classifier_node(
        &self,
        input: &str,
        classes: usize,
        namespace: &str,
    ) -> SKOnnxNode {
        let mut node = SKOnnxNode::ml(
            "TreeEnsembleClassifier",
            vec![input.to_string()],
            vec![
                format!("{namespace}tree_label"),
                format!("{namespace}probabilities"),
            ],
        );
        node = self.push_common_attributes(node);
        node.attribute_integers("class_treeids", self.class_treeids.clone())
            .attribute_integers("class_nodeids", self.class_nodeids.clone())
            .attribute_integers("class_ids", self.class_ids.clone())
            .attribute_floats("class_weights", self.class_weights.clone())
            .attribute_integers("classlabels_int64s", (0..classes as i64).collect())
            .attribute_string("post_transform", "NONE")
    }

    /// Build the regressor node over the accumulated attributes.
    pub(crate) fn regression_node(&self, input: &str, output: &str) -> SKOnnxNode {
        let node = SKOnnxNode::ml(
            "TreeEnsembleRegressor",
            vec![input.to_string()],
            vec![output.to_string()],
        );
        self.push_common_attributes(node)
            .attribute_integers("target_treeids", self.target_treeids.clone())
            .attribute_integers("target_nodeids", self.target_nodeids.clone())
            .attribute_integers("target_ids", vec![0; self.target_treeids.len()])
            .attribute_floats("target_weights", self.target_weights.clone())
            .attribute_integer("n_targets", 1)
            .attribute_string("aggregate_function", "SUM")
            .attribute_string("post_transform", "NONE")
    }

    fn push_common_attributes(&self, node: SKOnnxNode) -> SKOnnxNode {
        node.attribute_integers("nodes_treeids", self.nodes_treeids.clone())
            .attribute_integers("nodes_nodeids", self.nodes_nodeids.clone())
            .attribute_integers("nodes_featureids", self.nodes_featureids.clone())
            .attribute_strings("nodes_modes", self.nodes_modes.clone())
            .attribute_floats("nodes_values", self.nodes_values.clone())
            .attribute_integers("nodes_truenodeids", self.nodes_truenodeids.clone())
            .attribute_integers("nodes_falsenodeids", self.nodes_falsenodeids.clone())
            .attribute_integers(
                "nodes_missing_value_tracks_true",
                vec![0; self.nodes_treeids.len()],
            )
    }
}
