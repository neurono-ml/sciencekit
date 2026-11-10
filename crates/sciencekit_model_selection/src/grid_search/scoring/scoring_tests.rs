//! Hand-computed scoring policy checks (spec `grid-search`, requirement
//! `Score aggregation and best selection`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKDataView, SKTargetView};
use sciencekit_metrics::classification::SKAveraging;

use super::{SKClassifierScoring, SKRegressorScoring};
use crate::grid_search::evaluation::SKFoldScorer;
use crate::grid_search::mock_estimators::{SKMockFixedClassifierModel, SKMockFixedRegressorModel};

const TOLERANCE: f64 = 1e-12;

fn sk_features(number_of_rows: usize) -> Array2<f64> {
    Array2::zeros((number_of_rows, 1))
}

fn classifier_model() -> SKMockFixedClassifierModel {
    SKMockFixedClassifierModel::new(
        vec![0, 1, 0, 0],
        vec![[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]],
    )
}

fn score_classifier(policy: SKClassifierScoring) -> f64 {
    let features = sk_features(4);
    let truth = Array1::from_vec(vec![0_i64, 1, 1, 0]);
    policy
        .score(
            &classifier_model(),
            SKDataView::Dense(features.view()),
            SKTargetView::Integer(truth.view()),
        )
        .expect("the fold scores")
}

#[test]
fn grid_scoring_accuracy_matches_the_hand_computed_value() {
    assert!((score_classifier(SKClassifierScoring::Accuracy) - 0.75).abs() < TOLERANCE);
}

#[test]
fn grid_scoring_balanced_accuracy_matches_the_hand_computed_value() {
    assert!((score_classifier(SKClassifierScoring::BalancedAccuracy) - 0.75).abs() < TOLERANCE);
}

#[test]
fn grid_scoring_precision_recall_and_f1_match_the_hand_computed_values() {
    let binary = SKAveraging::Binary { positive_label: 1 };
    let precision = score_classifier(SKClassifierScoring::Precision(binary));
    let recall = score_classifier(SKClassifierScoring::Recall(binary));
    let f1 = score_classifier(SKClassifierScoring::F1(binary));
    assert!((precision - 1.0).abs() < TOLERANCE);
    assert!((recall - 0.5).abs() < TOLERANCE);
    assert!((f1 - 2.0 / 3.0).abs() < TOLERANCE);
}

#[test]
fn grid_scoring_auc_matches_the_hand_computed_value() {
    let score = score_classifier(SKClassifierScoring::Auc { positive_label: 1 });
    assert!((score - 1.0).abs() < TOLERANCE);
}

#[test]
fn grid_scoring_log_loss_matches_the_hand_computed_value() {
    let features = sk_features(2);
    let truth = Array1::from_vec(vec![0_i64, 1]);
    let model = SKMockFixedClassifierModel::new(vec![0, 1], vec![[0.8, 0.2], [0.1, 0.9]]);
    let score = SKClassifierScoring::LogLoss
        .score(
            &model,
            SKDataView::Dense(features.view()),
            SKTargetView::Integer(truth.view()),
        )
        .unwrap();
    // Selection maximises, so the scorer returns the negated log loss.
    let loss = -((0.8_f64).ln() + (0.9_f64).ln()) / 2.0;
    assert!((score + loss).abs() < TOLERANCE);
}

fn score_regressor(policy: SKRegressorScoring) -> f64 {
    let features = sk_features(3);
    let truth = array![1.0, 2.0, 3.0];
    let model = SKMockFixedRegressorModel::new(vec![1.0, 2.0, 4.0]);
    policy
        .score(
            &model,
            SKDataView::Dense(features.view()),
            SKTargetView::Continuous(truth.view()),
        )
        .expect("the fold scores")
}

#[test]
fn grid_scoring_negative_mean_squared_error_is_negated() {
    let score = score_regressor(SKRegressorScoring::NegativeMeanSquaredError);
    assert!((score + 1.0 / 3.0).abs() < TOLERANCE);
}

#[test]
fn grid_scoring_negative_mean_absolute_error_is_negated() {
    let score = score_regressor(SKRegressorScoring::NegativeMeanAbsoluteError);
    assert!((score + 1.0 / 3.0).abs() < TOLERANCE);
}

#[test]
fn grid_scoring_r2_matches_the_hand_computed_value() {
    assert!((score_regressor(SKRegressorScoring::R2) - 0.5).abs() < TOLERANCE);
}

#[test]
fn grid_scoring_requires_probabilities_only_for_ranking_policies() {
    assert!(!SKClassifierScoring::Accuracy.requires_probabilities());
    assert!(SKClassifierScoring::LogLoss.requires_probabilities());
    assert!(SKClassifierScoring::Auc { positive_label: 1 }.requires_probabilities());
}
