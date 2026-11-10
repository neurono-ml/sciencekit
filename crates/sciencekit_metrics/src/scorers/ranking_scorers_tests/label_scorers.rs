//! Label-based ranking adapters: balanced accuracy and MCC.

use ndarray::array;
use sciencekit_common::{SKLabelScorer, SKTargetView};

use super::fixtures::PanickingClassifier;
use crate::ranking::{sk_balanced_accuracy_score, sk_matthews_correlation_coefficient};
use crate::scorers::{SKBalancedAccuracyScorer, SKMatthewsCorrelationScorer};

/// The balanced-accuracy adapter equals the free function.
#[test]
fn ranking_scorers_balanced_accuracy_equals_free_function() {
    let truth_labels = array![0_i64, 0, 0, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let labels = array![0_i64, 0, 1, 1, 0];
    let adapter =
        <SKBalancedAccuracyScorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
            &SKBalancedAccuracyScorer,
            truth,
            labels.view(),
        )
        .expect("valid input");
    let expected =
        sk_balanced_accuracy_score::<f64>(&[0, 0, 0, 1, 1], &[0, 0, 1, 1, 0]).expect("valid input");
    assert!((adapter - expected).abs() < 1e-12);
}

/// The MCC adapter equals the free function.
#[test]
fn ranking_scorers_matthews_equals_free_function() {
    let truth_labels = array![0_i64, 0, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let labels = array![0_i64, 0, 1, 0];
    let adapter = <SKMatthewsCorrelationScorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
        &SKMatthewsCorrelationScorer,
        truth,
        labels.view(),
    )
    .expect("valid input");
    let expected = sk_matthews_correlation_coefficient::<f64>(&[0, 0, 1, 1], &[0, 0, 1, 0])
        .expect("valid input");
    assert!((adapter - expected).abs() < 1e-12);
}
