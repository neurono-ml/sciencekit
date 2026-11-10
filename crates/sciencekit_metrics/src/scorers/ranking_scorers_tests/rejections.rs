//! Rejected configurations of the ranking adapter surface.

use ndarray::array;
use sciencekit_common::{SKError, SKProbabilityScorer, SKTargetView};

use super::fixtures::PanickingClassifier;
use crate::scorers::SKAucScorer;

/// A multiclass probability matrix is outside the binary ranking adapter.
#[test]
fn ranking_scorers_auc_rejects_multiclass_probabilities() {
    let truth_labels = array![0_i64, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.6_f64, 0.3, 0.1], [0.2, 0.5, 0.3]];
    let error =
        <SKAucScorer as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
            &SKAucScorer::new(1),
            truth,
            probabilities.view(),
        )
        .expect_err("three columns are not a binary problem");
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}

/// A positive label absent from an explicit class list is rejected.
#[test]
fn ranking_scorers_auc_rejects_unknown_positive_label() {
    let truth_labels = array![0_i64, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.6_f64, 0.4], [0.2, 0.8]];
    let error =
        <SKAucScorer as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
            &SKAucScorer::new(7).with_class_labels(&[0, 1]),
            truth,
            probabilities.view(),
        )
        .expect_err("label 7 is not a class");
    assert!(matches!(error, SKError::UnknownLabel { label: 7 }));
}
