//! Companion tests for early stopping: seeded validation splits, best-stage
//! restore and fitted stage-count reporting (task 3.3).

use std::collections::HashSet;

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sciencekit_tree::SKMaximumFeatures;

use super::early_stopping::{
    SKEarlyStoppingConfig, SKEarlyStoppingMonitor, sk_split_classification_validation_rows,
    sk_split_regression_validation_rows,
};
use super::loss::{SKMulticlassLogLoss, SKSquaredErrorLoss};
use super::multiclass_loop::sk_run_multiclass_boosting;
use super::scalar_loop::{sk_run_binary_boosting, sk_run_scalar_boosting};
use super::stage::SKBoostingStageParameters;

fn stage_parameters(learning_rate: f64, maximum_depth: usize) -> SKBoostingStageParameters<f64> {
    SKBoostingStageParameters {
        learning_rate,
        maximum_depth,
        minimum_samples_leaf: 1,
        maximum_features: SKMaximumFeatures::All,
        subsample: 1.0,
        l2_regularization: 0.0,
        random_seed: 17,
    }
}

/// Seeded regression and stratified classification validation splits partition
/// `0..rows` deterministically and keep every class in the training set.
#[test]
fn seeded_validation_splits_are_deterministic_disjoint_and_stratified() {
    let (training_first, validation_first) =
        sk_split_regression_validation_rows(30, 0.2_f64, 42).expect("split");
    let (training_second, validation_second) =
        sk_split_regression_validation_rows(30, 0.2_f64, 42).expect("split");
    let (_, validation_other) =
        sk_split_regression_validation_rows(30, 0.2_f64, 99).expect("split");

    assert_eq!(training_first, training_second);
    assert_eq!(validation_first, validation_second);
    assert_ne!(validation_first, validation_other);
    assert_eq!(validation_first.len(), 6);
    assert_eq!(training_first.len(), 24);

    let union: HashSet<usize> = training_first
        .iter()
        .chain(validation_first.iter())
        .copied()
        .collect();
    assert_eq!(union.len(), 30);

    let classes = [0usize, 0, 1, 1, 2, 2];
    let (class_training, class_validation) =
        sk_split_classification_validation_rows(&classes, 3, 0.2_f64, 7).expect("stratified split");
    assert!(!class_validation.is_empty());
    let training_classes: HashSet<usize> = class_training.iter().map(|&row| classes[row]).collect();
    assert_eq!(training_classes.len(), 3);
}

/// Monitor restores the stage with minimal validation loss when later stages
/// plateau or degrade.
#[test]
fn monitor_stops_after_patience_and_reports_best_stage_count() {
    let mut monitor = SKEarlyStoppingMonitor::new(3, 1e-4_f64);
    assert!(!monitor.record(1, 1.00));
    assert!(!monitor.record(2, 0.50));
    assert!(!monitor.record(3, 0.50005));
    assert!(!monitor.record(4, 0.55));
    assert!(monitor.record(5, 0.60));
    assert_eq!(monitor.best_stage_count(), 2);
}

/// Scenario: Early stop reports fewer stages and restores the best stage count,
/// while disabled early stopping fits every configured stage.
#[test]
fn scalar_binary_and_multiclass_loops_stop_early_and_restore_best_stages() {
    let features = Array2::from_shape_fn((40, 2), |(row, column)| {
        if column == 0 {
            if row < 20 { 0.0 } else { 10.0 }
        } else {
            row as f64 * 0.1
        }
    });
    let targets: Vec<f64> = (0..40)
        .map(|row| if row < 20 { -3.0 } else { 5.0 })
        .collect();
    let view = SKDataView::Dense(features.view());

    let early = sk_run_scalar_boosting(
        &SKSquaredErrorLoss::new(),
        &view,
        &targets,
        50,
        &stage_parameters(1.0, 1),
        Some(SKEarlyStoppingConfig {
            validation_fraction: 0.2,
            patience: 3,
            tolerance: 1e-4,
            random_seed: 17,
        }),
    )
    .expect("early-stopped scalar boosting succeeds");
    assert_eq!(early.stages.len(), 1);
    assert!(early.initial_prediction.is_finite());

    let full = sk_run_scalar_boosting(
        &SKSquaredErrorLoss::new(),
        &view,
        &targets,
        12,
        &stage_parameters(0.2, 1),
        None,
    )
    .expect("unstopped scalar boosting fits all stages");
    assert_eq!(full.stages.len(), 12);
    assert!((full.initial_prediction - 1.0).abs() < 1e-12);

    let binary_indices: Vec<usize> = (0..40).map(|row| usize::from(row >= 20)).collect();
    let binary_targets: Vec<f64> = binary_indices.iter().map(|&class| class as f64).collect();
    let binary_early = sk_run_binary_boosting(
        &super::loss::SKBinaryLogLoss::new(),
        &view,
        &binary_indices,
        &binary_targets,
        50,
        &stage_parameters(4.0, 1),
        Some(SKEarlyStoppingConfig {
            validation_fraction: 0.2,
            patience: 3,
            tolerance: 1e-2,
            random_seed: 17,
        }),
    )
    .expect("binary early stop fits");
    assert!(binary_early.stages.len() < 50);

    let multiclass_features = Array2::from_shape_fn((45, 2), |(row, column)| {
        let cluster = (row % 3) as f64 * 10.0;
        cluster + column as f64 * 0.1
    });
    let class_indices: Vec<usize> = (0..45).map(|row| row % 3).collect();
    let multiclass_loss = SKMulticlassLogLoss::build()
        .number_of_classes(3)
        .build()
        .expect("builds");
    let multiclass_early = sk_run_multiclass_boosting(
        &multiclass_loss,
        &SKDataView::Dense(multiclass_features.view()),
        &class_indices,
        60,
        &stage_parameters(1.5, 2),
        Some(SKEarlyStoppingConfig {
            validation_fraction: 0.2,
            patience: 3,
            tolerance: 1e-3,
            random_seed: 17,
        }),
    )
    .expect("multiclass early stop fits");
    assert!(multiclass_early.stages.len() < 60);
    assert!(!multiclass_early.stages.is_empty());
    assert_eq!(multiclass_early.initial_predictions.len(), 3);
}
