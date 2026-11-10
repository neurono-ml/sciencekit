//! The pure and convenient forms of the probability scorer.

use ndarray::{Array2, array};

use super::fixtures::{ArgmaxAccuracy, ConstantProbabilityClassifier, PanickingClassifier};
use crate::SKProbabilityScorer;
use crate::{SKDataView, SKTargetView};

/// The pure form scores stored probabilities without running inference.
#[test]
fn probability_pure_form_does_not_infer() {
    let truth_labels = array![0_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.9_f64, 0.1], [0.2, 0.8], [0.3, 0.7]];
    let score =
        <ArgmaxAccuracy as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
            &ArgmaxAccuracy,
            truth,
            probabilities.view(),
        )
        .expect("valid input");
    assert!((score - 1.0).abs() < 1e-12);
}

/// The constant classifier scores perfectly when every target is its class.
#[test]
fn probability_constant_classifier_scores_perfectly() {
    let model = ConstantProbabilityClassifier;
    let features = Array2::<f64>::zeros((3, 2));
    let truth_labels = array![1_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let score = ArgmaxAccuracy
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect("valid input");
    assert!((score - 1.0).abs() < 1e-12);
}

/// The convenient form equals the pure form over the model's probabilities.
#[test]
fn probability_convenient_form_equals_pure_form() {
    let model = ConstantProbabilityClassifier;
    let features = Array2::<f64>::zeros((3, 2));
    let truth_labels = array![1_i64, 0, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let convenient = ArgmaxAccuracy
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect("valid input");
    let probabilities = array![[0.0_f64, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let pure =
        <ArgmaxAccuracy as SKProbabilityScorer<f64, ConstantProbabilityClassifier>>::score_from_probabilities(
            &ArgmaxAccuracy,
            truth,
            probabilities.view(),
        )
        .expect("valid input");
    assert!((convenient - pure).abs() < 1e-12);
}
