//! Stage-wise tree growth, Newton leaf replacement and shrunk raw-score updates.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_RANDOM_DOMAIN_STAGE, SK_RANDOM_DOMAIN_TREE, SKDataView, SKError, SKFloat, sk_derive_seed,
};
use sciencekit_tree::{SKMaximumFeatures, SKStructureConstraints, SKTree, SKTreeStructure};

use super::init::sk_newton_leaf_value_from_negative_gradients;
use super::loss::{SKGradientLoss, SKMulticlassLogLoss};
use super::scores::sk_update_raw_scores;

/// Validated per-stage parameters shared across regression and classification.
#[derive(Debug, Clone)]
pub(crate) struct SKBoostingStageParameters<F> {
    pub(crate) learning_rate: F,
    pub(crate) maximum_depth: usize,
    pub(crate) minimum_samples_leaf: usize,
    pub(crate) maximum_features: SKMaximumFeatures<F>,
    pub(crate) subsample: F,
    pub(crate) l2_regularization: F,
    pub(crate) random_seed: u64,
}

/// One fitted boosting tree together with its structure importances and sample
/// count.
#[derive(Debug, Clone)]
pub(crate) struct SKFittedStageTree<F> {
    pub(crate) tree: SKTree<F>,
    pub(crate) feature_importances: Vec<F>,
    pub(crate) sample_count: usize,
}

/// Derive the deterministic split seed for `(stage_index, class_index)`.
pub(crate) fn sk_stage_tree_seed(global_seed: u64, stage_index: usize, class_index: usize) -> u64 {
    let stage_seed = sk_derive_seed(global_seed, SK_RANDOM_DOMAIN_STAGE, stage_index as u64);
    sk_derive_seed(stage_seed, SK_RANDOM_DOMAIN_TREE, class_index as u64)
}

/// Fit one tree on `(negative_gradients, hessians)` and replace its leaf values
/// with the unshrunk Newton steps.
pub(crate) fn sk_fit_stage_tree<F: SKFloat>(
    features: &SKDataView<'_, F>,
    negative_gradients: &[F],
    hessians: &[F],
    parameters: &SKBoostingStageParameters<F>,
    stage_index: usize,
    class_index: usize,
) -> Result<SKFittedStageTree<F>, SKError> {
    let _ = parameters.subsample;
    let tree_seed = sk_stage_tree_seed(parameters.random_seed, stage_index, class_index);
    let minimum_samples_split = 2usize.max(parameters.minimum_samples_leaf.saturating_mul(2));
    let constraints = SKStructureConstraints::<F>::build()
        .maximum_depth(parameters.maximum_depth)
        .minimum_samples_split(minimum_samples_split)
        .minimum_samples_leaf(parameters.minimum_samples_leaf)
        .maximum_features(parameters.maximum_features.clone())
        .random_seed(tree_seed)
        .build()?;
    let residual_array = Array1::from_vec(negative_gradients.to_vec());
    let mut structure = match features {
        SKDataView::Dense(dense) => {
            SKTreeStructure::build_structure(*dense, residual_array.view(), &constraints)?
        }
        SKDataView::Sparse(sparse) => {
            SKTreeStructure::build_structure(*sparse, residual_array.view(), &constraints)?
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            });
        }
    };
    let leaf_values: Vec<F> = (0..structure.leaf_count())
        .map(|ordinal| {
            let rows = structure
                .leaf_sample_indices(ordinal)
                .expect("leaf ordinal in bounds");
            let leaf_negative_gradients: Vec<F> =
                rows.iter().map(|&row| negative_gradients[row]).collect();
            let leaf_hessians: Vec<F> = rows.iter().map(|&row| hessians[row]).collect();
            sk_newton_leaf_value_from_negative_gradients(
                &leaf_negative_gradients,
                &leaf_hessians,
                parameters.l2_regularization,
            )
        })
        .collect();
    structure.replace_leaf_values(&leaf_values)?;
    let feature_importances = structure.feature_importances().to_vec();
    let sample_count = structure.sample_count();
    Ok(SKFittedStageTree {
        tree: structure.into_tree(),
        feature_importances,
        sample_count,
    })
}

/// Execute one scalar boosting stage: compute negative gradients and Hessians,
/// fit a Newton-leaf tree and update `raw_scores` by `learning_rate * tree(x)`.
pub(crate) fn sk_fit_scalar_stage<F: SKFloat, L: SKGradientLoss<F>>(
    loss: &L,
    features: &SKDataView<'_, F>,
    targets: &[F],
    raw_scores: &mut [F],
    parameters: &SKBoostingStageParameters<F>,
    stage_index: usize,
) -> Result<SKFittedStageTree<F>, SKError> {
    let negative_gradients: Vec<F> = targets
        .iter()
        .zip(raw_scores.iter())
        .map(|(&target, &score)| loss.negative_gradient(target, score))
        .collect();
    let hessians: Vec<F> = targets
        .iter()
        .zip(raw_scores.iter())
        .map(|(&target, &score)| loss.hessian(target, score))
        .collect();
    let fitted = sk_fit_stage_tree(
        features,
        &negative_gradients,
        &hessians,
        parameters,
        stage_index,
        0,
    )?;
    sk_update_raw_scores(features, &fitted.tree, parameters.learning_rate, raw_scores);
    Ok(fitted)
}

/// Execute one multiclass boosting stage (`C` trees, one per canonical class).
pub(crate) fn sk_fit_multiclass_stage<F: SKFloat>(
    loss: &SKMulticlassLogLoss,
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    raw_scores: &mut Array2<F>,
    parameters: &SKBoostingStageParameters<F>,
    stage_index: usize,
) -> Result<Vec<SKFittedStageTree<F>>, SKError> {
    let (negative_gradients, hessians) =
        loss.stage_gradients_and_hessians(class_indices, raw_scores);
    negative_gradients
        .iter()
        .zip(hessians.iter())
        .enumerate()
        .map(|(class_index, (class_gradients, class_hessians))| {
            let fitted = sk_fit_stage_tree(
                features,
                class_gradients,
                class_hessians,
                parameters,
                stage_index,
                class_index,
            )?;
            let mut column_scores: Vec<F> = raw_scores.column(class_index).to_vec();
            sk_update_raw_scores(
                features,
                &fitted.tree,
                parameters.learning_rate,
                &mut column_scores,
            );
            raw_scores
                .column_mut(class_index)
                .iter_mut()
                .zip(column_scores.iter())
                .for_each(|(destination, &updated)| *destination = updated);
            Ok(fitted)
        })
        .collect()
}
