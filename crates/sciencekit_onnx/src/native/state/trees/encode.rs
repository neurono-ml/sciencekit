//! Encoding a rebuilt tree (or forest) as neutral state tensors.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKFloat, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    sk_model_tensor_data_type,
};

use super::rebuild::Tree;
use crate::native::error::SKOnnxNativeError;

/// A single fitted tree, with its feature-importance vector.
pub(crate) fn single_tree<F: SKFloat>(
    tree: &Tree,
    features: usize,
    metadata: SKModelMetadata,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let mut tensors = tree_tensors::<F>(tree, "tree")?;
    tensors.push(
        SKTensor::floating(
            "tree.feature_importances",
            vec![features],
            vec![F::zero(); features],
        )
        .map_err(native)?,
    );
    SKModelState::new(metadata, tensors).map_err(native)
}

/// A forest regressor whose members are restored to their unscaled leaves.
pub(crate) fn forest_regressor<F: SKFloat>(
    trees: &[Tree],
    features: usize,
) -> Result<SKModelState<F>, SKOnnxNativeError> {
    let count = trees.len();
    let metadata = tree_metadata::<F>("SKRandomForestRegressor", features)?
        .with_parameter("number_of_trees", SKParameterValue::from(count as i64))
        .map_err(native)?;
    let mut tensors = vec![
        SKTensor::floating(
            "forest.feature_importances",
            vec![features],
            vec![F::zero(); features],
        )
        .map_err(native)?,
    ];
    for (index, tree) in trees.iter().enumerate() {
        let scaled = scale_tree(tree, count as f64);
        let prefix = format!("forest.tree.{index}.t.tree");
        tensors.extend(tree_tensors::<F>(&scaled, &prefix)?);
        // The embedded tree is imported as a real decision-tree model, whose
        // state carries its own (here unweighted) feature importances.
        tensors.push(
            SKTensor::floating(
                format!("{prefix}.feature_importances"),
                vec![features],
                vec![F::zero(); features],
            )
            .map_err(native)?,
        );
    }
    SKModelState::new(metadata, tensors).map_err(native)
}

/// The base metadata every tree estimator shares.
pub(crate) fn tree_metadata<F: SKFloat>(
    estimator_name: &str,
    features: usize,
) -> Result<SKModelMetadata, SKOnnxNativeError> {
    SKModelMetadata::new(
        estimator_name,
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        features,
    )
    .map_err(native)
}

fn scale_tree(tree: &Tree, factor: f64) -> Tree {
    Tree {
        split_feature: tree.split_feature.clone(),
        threshold: tree.threshold.clone(),
        left: tree.left.clone(),
        right: tree.right.clone(),
        values: tree
            .values
            .iter()
            .map(|values| values.iter().map(|value| value * factor).collect())
            .collect(),
    }
}

fn tree_tensors<F: SKFloat>(
    tree: &Tree,
    prefix: &str,
) -> Result<Vec<SKTensor<F>>, SKOnnxNativeError> {
    let nodes = tree.split_feature.len();
    let mut flat = Vec::new();
    let mut offsets = Vec::with_capacity(nodes + 1);
    offsets.push(0_i64);
    for values in &tree.values {
        flat.extend(
            values
                .iter()
                .map(|value| F::from(*value).unwrap_or_else(F::zero)),
        );
        offsets.push(flat.len() as i64);
    }
    let floating = |name: &str, values: Vec<F>| {
        SKTensor::floating(format!("{prefix}.nodes.{name}"), vec![nodes], values).map_err(native)
    };
    Ok(vec![
        SKTensor::integer(
            format!("{prefix}.nodes.split_feature"),
            vec![nodes],
            tree.split_feature.clone(),
        )
        .map_err(native)?,
        floating(
            "threshold",
            tree.threshold
                .iter()
                .map(|value| F::from(*value).unwrap_or_else(F::zero))
                .collect(),
        )?,
        SKTensor::integer(
            format!("{prefix}.nodes.left"),
            vec![nodes],
            tree.left.clone(),
        )
        .map_err(native)?,
        SKTensor::integer(
            format!("{prefix}.nodes.right"),
            vec![nodes],
            tree.right.clone(),
        )
        .map_err(native)?,
        SKTensor::integer(
            format!("{prefix}.nodes.sample_count"),
            vec![nodes],
            vec![0; nodes],
        )
        .map_err(native)?,
        floating("impurity", vec![F::zero(); nodes])?,
        SKTensor::floating(format!("{prefix}.nodes.value"), vec![flat.len()], flat)
            .map_err(native)?,
        SKTensor::integer(
            format!("{prefix}.nodes.value_offsets"),
            vec![nodes + 1],
            offsets,
        )
        .map_err(native)?,
    ])
}

fn native(error: sciencekit_common::SKError) -> SKOnnxNativeError {
    SKOnnxNativeError::Native(sciencekit_interop::native::SKNativeImportError::from(error))
}
