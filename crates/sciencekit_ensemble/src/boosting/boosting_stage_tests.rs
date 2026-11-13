//! Companion tests for Newton leaf values, shrinkage and single-stage updates
//! (task 3.1).

use ndarray::array;
use sciencekit_common::SKDataView;
use sciencekit_tree::SKMaximumFeatures;

use super::init::sk_newton_leaf_value;
use super::loss::{SKBinaryLogLoss, SKSquaredErrorLoss};
use super::stage::{SKBoostingStageParameters, sk_fit_scalar_stage};

fn default_stage_parameters(
    maximum_depth: usize,
    learning_rate: f64,
) -> SKBoostingStageParameters<f64> {
    SKBoostingStageParameters {
        learning_rate,
        maximum_depth,
        minimum_samples_leaf: 1,
        maximum_features: SKMaximumFeatures::All,
        subsample: 1.0,
        l2_regularization: 0.0,
        random_seed: 0,
    }
}

/// Scenario: One-leaf Newton step (`g = [2, -1]`, `h = [1, 1]`, `λ = 0` -> `-0.5`),
/// L2 damping, and zero-Hessian degeneracy guard.
#[test]
fn newton_leaf_value_matches_hand_computed_step_and_degeneracy_guard() {
    let step = sk_newton_leaf_value(&[2.0_f64, -1.0], &[1.0, 1.0], 0.0);
    assert!((step - (-0.5)).abs() < 1e-12);

    let damped = sk_newton_leaf_value(&[2.0_f64, -1.0], &[1.0, 1.0], 2.0);
    assert!((damped - (-0.25)).abs() < 1e-12);

    let degenerate = sk_newton_leaf_value(&[3.0_f64, 1.0], &[0.0, 0.0], 0.0);
    assert_eq!(degenerate, 0.0);
}

/// A single depth-0 stage on `y = [0, 3]` starting at raw score `2.0` has
/// gradients `g = [2, -1]` (negative gradients `[-2, 1]`), Hessians `[1, 1]`,
/// unshrunk Newton leaf `-0.5`, and after `learning_rate = 0.1` updates raw
/// scores to `1.95`.
#[test]
fn single_stage_stump_and_split_apply_newton_leaves_and_shrinkage() {
    let features = array![[0.0], [1.0]];
    let view = SKDataView::Dense(features.view());
    let targets = [0.0_f64, 3.0];
    let mut raw_scores = vec![2.0_f64, 2.0];
    let parameters = default_stage_parameters(0, 0.1);

    let fitted = sk_fit_scalar_stage(
        &SKSquaredErrorLoss::new(),
        &view,
        &targets,
        &mut raw_scores,
        &parameters,
        0,
    )
    .expect("stage fits");

    assert_eq!(fitted.tree.leaf_count(), 1);
    assert!((fitted.tree.leaf_value_for(|_| 0.0) - (-0.5)).abs() < 1e-12);
    assert!((raw_scores[0] - 1.95).abs() < 1e-12);
    assert!((raw_scores[1] - 1.95).abs() < 1e-12);

    // Binary log-loss depth-1 stage on [0, 0, 1, 1] starting at raw = 0 (p = 0.5):
    // Left leaf has neg_grad = [-0.5, -0.5], h = [0.25, 0.25] -> Newton = -1.0 / 0.5 = -2.0.
    // Right leaf has neg_grad = [0.5, 0.5], h = [0.25, 0.25] -> Newton = +1.0 / 0.5 = +2.0.
    // With learning_rate = 0.25, raw scores update to [-0.5, -0.5, +0.5, +0.5].
    let binary_features = array![[0.0], [1.0], [10.0], [11.0]];
    let binary_view = SKDataView::Dense(binary_features.view());
    let binary_targets = [0.0_f64, 0.0, 1.0, 1.0];
    let mut binary_scores = vec![0.0_f64; 4];
    let binary_stage = sk_fit_scalar_stage(
        &SKBinaryLogLoss::new(),
        &binary_view,
        &binary_targets,
        &mut binary_scores,
        &default_stage_parameters(1, 0.25),
        0,
    )
    .expect("binary stage fits");

    assert_eq!(binary_stage.tree.leaf_count(), 2);
    assert_eq!(binary_stage.sample_count, 4);
    assert_eq!(binary_stage.feature_importances.len(), 1);
    assert!((binary_stage.tree.leaf_value_for(|_| 0.5) - (-2.0)).abs() < 1e-12);
    assert!((binary_stage.tree.leaf_value_for(|_| 10.5) - 2.0).abs() < 1e-12);
    assert!((binary_scores[0] - (-0.5)).abs() < 1e-12);
    assert!((binary_scores[2] - 0.5).abs() < 1e-12);
}

/// Multiclass single-stage update fits `C` Newton-leaf trees and updates each
/// class column of the raw score matrix independently.
#[test]
fn multiclass_single_stage_fits_one_tree_per_class() {
    use super::init::sk_initialize_multiclass_scores;
    use super::loss::SKMulticlassLogLoss;
    use super::stage::sk_fit_multiclass_stage;

    let features = array![[0.0], [1.0], [5.0], [6.0], [10.0], [11.0]];
    let view = SKDataView::Dense(features.view());
    let class_indices = [0usize, 0, 1, 1, 2, 2];
    let loss = SKMulticlassLogLoss::build()
        .number_of_classes(3)
        .build()
        .expect("builds");
    let (_, mut raw_scores) = sk_initialize_multiclass_scores::<f64>(&loss, &class_indices);
    let trees = sk_fit_multiclass_stage(
        &loss,
        &view,
        &class_indices,
        &mut raw_scores,
        &default_stage_parameters(2, 0.2),
        0,
    )
    .expect("multiclass stage fits");

    assert_eq!(trees.len(), 3);
    assert!(raw_scores[[0, 0]] > raw_scores[[0, 1]]);
    assert!(raw_scores[[2, 1]] > raw_scores[[2, 2]]);
    assert!(raw_scores[[4, 2]] > raw_scores[[4, 0]]);
}
