//! Companion tests for stochastic row subsampling and feature subsampling
//! (task 3.2).

use std::collections::HashSet;

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sciencekit_tree::SKMaximumFeatures;

use super::loss::SKSquaredErrorLoss;
use super::stage::{SKBoostingStageParameters, sk_fit_scalar_stage};
use super::subsample::sk_select_stage_rows;

/// Scenario: Full subsample uses every row (`subsample = 1.0`).
#[test]
fn full_subsample_uses_every_training_row() {
    let selected = sk_select_stage_rows(20, 1.0_f64, 42, 0, 0);
    assert_eq!(selected, (0..20).collect::<Vec<_>>());

    let features = Array2::from_shape_fn((20, 3), |(row, col)| (row * 3 + col) as f64);
    let targets: Vec<f64> = (0..20).map(|row| row as f64 * 2.0).collect();
    let mut raw_scores = vec![10.0_f64; 20];
    let parameters = SKBoostingStageParameters {
        learning_rate: 0.1,
        maximum_depth: 2,
        minimum_samples_leaf: 1,
        maximum_features: SKMaximumFeatures::All,
        subsample: 1.0,
        l2_regularization: 0.0,
        random_seed: 7,
    };
    let fitted = sk_fit_scalar_stage(
        &SKSquaredErrorLoss::new(),
        &SKDataView::Dense(features.view()),
        &targets,
        &mut raw_scores,
        &parameters,
        0,
    )
    .expect("full-sample stage fits");
    assert_eq!(fitted.sample_count, 20);
}

/// Scenario: Subsample is reproducible across repeated calls for both row
/// subsets (`subsample < 1`) and feature subsets (`maximum_features`), while
/// diversifying across stages.
#[test]
fn stochastic_row_and_feature_subsampling_are_reproducible_and_diversify() {
    let first_draw = sk_select_stage_rows(40, 0.5_f64, 2026, 3, 0);
    let second_draw = sk_select_stage_rows(40, 0.5_f64, 2026, 3, 0);
    let next_stage_draw = sk_select_stage_rows(40, 0.5_f64, 2026, 4, 0);
    assert_eq!(first_draw.len(), 20);
    assert_eq!(first_draw.iter().copied().collect::<HashSet<_>>().len(), 20);
    assert_eq!(first_draw, second_draw);
    assert_ne!(first_draw, next_stage_draw);

    let features = Array2::from_shape_fn((60, 9), |(row, column)| {
        if column == 8 {
            if row < 30 { 0.0 } else { 10.0 }
        } else {
            ((row * 5 + column * 7) % 13) as f64
        }
    });
    let targets: Vec<f64> = (0..60)
        .map(|row| if row < 30 { -2.0 } else { 4.0 })
        .collect();
    let parameters = SKBoostingStageParameters {
        learning_rate: 0.1,
        maximum_depth: 1,
        minimum_samples_leaf: 1,
        maximum_features: SKMaximumFeatures::Sqrt,
        subsample: 0.5,
        l2_regularization: 0.0,
        random_seed: 99,
    };

    let run_stages = || {
        let mut scores = vec![0.0_f64; 60];
        let trees: Vec<_> = (0..12)
            .map(|stage| {
                sk_fit_scalar_stage(
                    &SKSquaredErrorLoss::new(),
                    &SKDataView::Dense(features.view()),
                    &targets,
                    &mut scores,
                    &parameters,
                    stage,
                )
                .expect("stage fits")
            })
            .collect();
        (trees, scores)
    };

    let (first_trees, first_scores) = run_stages();
    let (second_trees, second_scores) = run_stages();
    assert!(first_trees.iter().all(|stage| stage.sample_count == 30));
    assert_eq!(first_scores, second_scores);
    assert!(
        first_trees
            .iter()
            .zip(second_trees.iter())
            .all(|(left, right)| left.tree == right.tree)
    );

    let root_features: HashSet<Option<usize>> = first_trees
        .iter()
        .map(|stage| stage.tree.node(0).and_then(|node| node.split_feature()))
        .collect();
    assert!(root_features.len() > 1);
}
