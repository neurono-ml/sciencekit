//! Probability-based ranking adapters: AUC, average precision and log loss.

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKProbabilityScorer, SKTargetView};

use super::fixtures::{FixedProbabilityClassifier, PanickingClassifier};
use crate::ranking::{sk_average_precision_score, sk_log_loss, sk_roc_auc_score};
use crate::scorers::{SKAucScorer, SKAveragePrecisionScorer, SKLogLossScorer};

/// The AUC adapter equals the free function.
#[test]
fn ranking_scorers_auc_equals_free_function() {
    let truth_labels = array![0_i64, 0, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.9_f64, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]];
    let scorer = SKAucScorer::new(1);
    let adapter =
        <SKAucScorer as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
            &scorer,
            truth,
            probabilities.view(),
        )
        .expect("valid input");
    let expected =
        sk_roc_auc_score(&[0.1_f64, 0.2, 0.7, 0.9], &[0, 0, 1, 1], 1).expect("valid input");
    assert!((adapter - expected).abs() < 1e-12);
}

/// The AUC adapter's convenient form delegates to the pure form.
#[test]
fn ranking_scorers_auc_convenient_form_delegates() {
    let model = FixedProbabilityClassifier {
        probabilities: vec![
            vec![0.9, 0.1],
            vec![0.8, 0.2],
            vec![0.3, 0.7],
            vec![0.1, 0.9],
        ],
    };
    let features = Array2::<f64>::zeros((4, 2));
    let truth_labels = array![0_i64, 0, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let convenient = SKAucScorer::new(1)
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect("valid input");
    assert!((convenient - 1.0).abs() < 1e-12);
}

/// The average-precision adapter equals the free function.
#[test]
fn ranking_scorers_average_precision_equals_free_function() {
    let truth_labels = array![0_i64, 0, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.9_f64, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]];
    let scorer = SKAveragePrecisionScorer::new(1);
    let adapter = <SKAveragePrecisionScorer as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
        &scorer,
        truth,
        probabilities.view(),
    )
    .expect("valid input");
    let expected = sk_average_precision_score(&[0.1_f64, 0.2, 0.7, 0.9], &[0, 0, 1, 1], 1)
        .expect("valid input");
    assert!((adapter - expected).abs() < 1e-12);
}

/// The log-loss adapter equals the free function.
#[test]
fn ranking_scorers_log_loss_equals_free_function() {
    let truth_labels = array![0_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.7_f64, 0.3], [0.4, 0.6], [0.2, 0.8]];
    let adapter = <SKLogLossScorer as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
        &SKLogLossScorer::new(),
        truth,
        probabilities.view(),
    )
    .expect("valid input");
    let expected = sk_log_loss(&[0_i64, 1, 1], probabilities.view(), None).expect("valid input");
    assert!((adapter - expected).abs() < 1e-12);
}

/// The log-loss adapter carries an explicit class-label list.
#[test]
fn ranking_scorers_log_loss_carries_class_labels() {
    let truth_labels = array![10_i64, 20, 20];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.7_f64, 0.3], [0.4, 0.6], [0.2, 0.8]];
    let adapter = <SKLogLossScorer as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
        &SKLogLossScorer::new().with_class_labels(&[10, 20]),
        truth,
        probabilities.view(),
    )
    .expect("valid input");
    let expected =
        sk_log_loss(&[10_i64, 20, 20], probabilities.view(), Some(&[10, 20])).expect("valid input");
    assert!((adapter - expected).abs() < 1e-12);
}
