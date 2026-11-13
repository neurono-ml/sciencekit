//! The `ai.onnx.ml` tree-ensemble operators.
//!
//! Trees are stored as flat parallel arrays; every row walks one tree from its
//! root to a leaf and accumulates the leaf weight. The arithmetic mirrors the
//! ONNX-ML definition so an imported prediction matches the exporter.

use ndarray::{ArrayD, IxDyn};
use serde_onnx::ir::Node;
use serde_onnx::ml::{OnnxOp, TreeEnsembleClassifier, TreeEnsembleRegressor};

use super::SKOnnxImportError;
use super::tree_walk::{distinct_trees, walk};
use super::value::SKOnnxValue;

/// Evaluate a `TreeEnsembleClassifier` node into `(labels, scores)`.
pub(crate) fn classifier(
    node: &Node,
    input: &SKOnnxValue,
) -> Result<Vec<SKOnnxValue>, SKOnnxImportError> {
    let ensemble = TreeEnsembleClassifier::from_node(node).map_err(invalid)?;
    let rows = matrix_rows(input, "TreeEnsembleClassifier")?;
    let labels =
        ensemble
            .classlabels_int64s
            .clone()
            .ok_or_else(|| SKOnnxImportError::InvalidOperator {
                operator: "TreeEnsembleClassifier".to_string(),
                detail: "string class labels are outside the supported subset".to_string(),
            })?;
    let classes = labels.len();
    let weights = ensemble.class_weights.clone().unwrap_or_default();
    let mut scores = vec![0.0_f64; rows.len() * classes];
    for (row, features) in rows.iter().enumerate() {
        for tree in distinct_trees(&ensemble.nodes) {
            let leaf = walk(&ensemble.nodes, tree, features);
            for (position, weight) in weights.iter().enumerate() {
                if ensemble.class_treeids[position] == tree
                    && ensemble.class_nodeids[position] == leaf
                {
                    let class = ensemble.class_ids[position] as usize;
                    if class < classes {
                        scores[row * classes + class] += f64::from(*weight);
                    }
                }
            }
        }
    }
    add_base_values(&mut scores, classes, ensemble.base_values.as_deref());
    let label_values: Vec<i64> = scores
        .chunks(classes)
        .map(|row| labels[arg_max(row)])
        .collect();
    Ok(vec![
        SKOnnxValue::Int64(
            ArrayD::from_shape_vec(IxDyn(&[rows.len()]), label_values).map_err(|error| {
                SKOnnxImportError::InvalidOperator {
                    operator: "TreeEnsembleClassifier".to_string(),
                    detail: error.to_string(),
                }
            })?,
        ),
        float_matrix(input, &rows, classes, scores)?,
    ])
}

/// Evaluate a `TreeEnsembleRegressor` node into its score matrix.
pub(crate) fn regressor(
    node: &Node,
    input: &SKOnnxValue,
) -> Result<Vec<SKOnnxValue>, SKOnnxImportError> {
    let ensemble = TreeEnsembleRegressor::from_node(node).map_err(invalid)?;
    let rows = matrix_rows(input, "TreeEnsembleRegressor")?;
    let targets = ensemble
        .n_targets
        .map(|value| value.max(1) as usize)
        .unwrap_or_else(|| {
            ensemble
                .target_ids
                .iter()
                .max()
                .map(|value| *value as usize + 1)
                .unwrap_or(1)
        });
    let weights = ensemble.target_weights.clone().unwrap_or_default();
    let mut scores = vec![0.0_f64; rows.len() * targets];
    for (row, features) in rows.iter().enumerate() {
        for tree in distinct_trees(&ensemble.nodes) {
            let leaf = walk(&ensemble.nodes, tree, features);
            for (position, weight) in weights.iter().enumerate() {
                if ensemble.target_treeids[position] == tree
                    && ensemble.target_nodeids[position] == leaf
                {
                    let target = ensemble.target_ids[position] as usize;
                    if target < targets {
                        scores[row * targets + target] += f64::from(*weight);
                    }
                }
            }
        }
    }
    add_base_values(&mut scores, targets, ensemble.base_values.as_deref());
    Ok(vec![float_matrix(input, &rows, targets, scores)?])
}

fn float_matrix(
    input: &SKOnnxValue,
    rows: &[Vec<f64>],
    width: usize,
    scores: Vec<f64>,
) -> Result<SKOnnxValue, SKOnnxImportError> {
    SKOnnxValue::from_f64_values(input.element_type(), vec![rows.len(), width], scores)
}

fn add_base_values(scores: &mut [f64], width: usize, base: Option<&[f32]>) {
    let Some(base) = base else { return };
    for row in scores.chunks_mut(width) {
        for (position, slot) in row.iter_mut().enumerate() {
            *slot += f64::from(*base.get(position).unwrap_or(&0.0));
        }
    }
}

fn matrix_rows(input: &SKOnnxValue, operator: &str) -> Result<Vec<Vec<f64>>, SKOnnxImportError> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(SKOnnxImportError::ShapeMismatch {
            expected: vec![shape.first().copied().unwrap_or(0), 0],
            found: shape,
        });
    }
    let _ = operator;
    let values = input.to_f64_values();
    Ok(values.chunks(shape[1]).map(|row| row.to_vec()).collect())
}

fn arg_max(row: &[f64]) -> usize {
    let mut best = 0usize;
    for (position, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = position;
        }
    }
    best
}

fn invalid(error: serde_onnx::ml::OpError) -> SKOnnxImportError {
    SKOnnxImportError::InvalidOperator {
        operator: "ai.onnx.ml tree ensemble".to_string(),
        detail: error.to_string(),
    }
}
