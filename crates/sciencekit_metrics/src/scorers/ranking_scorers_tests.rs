//! TDD tests for the ranking scorer adapters (spec `ranking-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKLabelScorer, SKProbabilityScorer,
    SKTargetView,
};

use crate::ranking::{
    sk_average_precision_score, sk_balanced_accuracy_score, sk_log_loss,
    sk_matthews_correlation_coefficient, sk_roc_auc_score,
};

use super::{
    SKAucScorer, SKAveragePrecisionScorer, SKBalancedAccuracyScorer, SKLogLossScorer,
    SKMatthewsCorrelationScorer,
};

/// A classifier returning a fixed probability matrix, ignoring the features.
struct FixedProbabilityClassifier {
    probabilities: Vec<Vec<f64>>,
}

impl<F: SKFloat> SKClassifierPredictor<F> for FixedProbabilityClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array1::zeros(self.probabilities.len()))
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let rows = self.probabilities.len();
        let columns = self.probabilities.first().map(Vec::len).unwrap_or(0);
        let mut output = Array2::zeros((rows, columns));
        for (row, values) in self.probabilities.iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                output[[row, column]] =
                    num_traits::cast::<f64, F>(*value).expect("values fit the scalar");
            }
        }
        Ok(output)
    }
}

/// A classifier whose inference must never run in the pure form.
struct PanickingClassifier;

impl<F: SKFloat> SKClassifierPredictor<F> for PanickingClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }
}

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
