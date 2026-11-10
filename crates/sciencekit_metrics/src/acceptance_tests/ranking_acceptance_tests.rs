//! Acceptance tests for the ranking metrics (PRD §8.7 / §10.3).
//!
//! Covers large data, f32/f64 parity, concurrent calls and the thread-safety
//! of every public ranking type.

use rayon::prelude::*;
use sciencekit_common::SKError;

use crate::ranking::{
    SKAucAccumulator, SKPrecisionRecallCurve, SKRankingAveraging, SKRocCurve,
    sk_average_precision_score, sk_balanced_accuracy_score, sk_log_loss,
    sk_matthews_correlation_coefficient, sk_multiclass_roc_auc_score, sk_roc_auc_score,
};
use crate::scorers::{
    SKAucScorer, SKAveragePrecisionScorer, SKBalancedAccuracyScorer, SKLogLossScorer,
    SKMatthewsCorrelationScorer,
};

const OBSERVATIONS: usize = 100_000;

/// A separable large dataset: the label is the thresholded score.
fn large_scores_and_labels() -> (Vec<f64>, Vec<i64>) {
    let scores: Vec<f64> = (0..OBSERVATIONS)
        .map(|index| index as f64 / OBSERVATIONS as f64)
        .collect();
    let labels: Vec<i64> = scores
        .iter()
        .map(|score| if *score >= 0.5 { 1 } else { 0 })
        .collect();
    (scores, labels)
}

/// Large, perfectly separable data scores one on every rank-based metric.
#[test]
fn ranking_acceptance_large_data_scores_one() {
    let (scores, labels) = large_scores_and_labels();
    let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert!((auc - 1.0).abs() < 1e-12);
    let average_precision = sk_average_precision_score(&scores, &labels, 1).expect("valid input");
    assert!((average_precision - 1.0).abs() < 1e-12);
    let predicted: Vec<i64> = scores
        .iter()
        .map(|score| if *score >= 0.5 { 1 } else { 0 })
        .collect();
    assert!((sk_balanced_accuracy_score::<f64>(&labels, &predicted).unwrap() - 1.0).abs() < 1e-12);
    assert!(
        (sk_matthews_correlation_coefficient::<f64>(&labels, &predicted).unwrap() - 1.0).abs()
            < 1e-12
    );
}

/// A large probability matrix produces a finite log loss.
#[test]
fn ranking_acceptance_large_data_log_loss_is_finite() {
    let (scores, labels) = large_scores_and_labels();
    let mut probabilities = ndarray::Array2::<f64>::zeros((OBSERVATIONS, 2));
    for (row, score) in scores.iter().enumerate() {
        let positive = 0.6 + 0.3 * score;
        probabilities[[row, 0]] = 1.0 - positive;
        probabilities[[row, 1]] = positive;
    }
    let loss = sk_log_loss(&labels, probabilities.view(), None).expect("valid input");
    assert!(loss.is_finite() && loss > 0.0 && loss < 1.0);
}

/// A large accumulator in batches equals the whole-input AUC.
#[test]
fn ranking_acceptance_large_accumulator_matches_whole_input() {
    let (scores, labels) = large_scores_and_labels();
    let mut accumulator = SKAucAccumulator::new();
    for (score, label) in scores.iter().zip(labels.iter()) {
        accumulator.observe(*score, *label);
    }
    let accumulated = accumulator.finalize(1).expect("valid input");
    let whole = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    assert_eq!(accumulated, whole);
}

/// f32 and f64 agree within the documented single-precision tolerance.
#[test]
fn ranking_acceptance_f32_matches_f64() {
    let scores_f64 = [0.1_f64, 0.4, 0.35, 0.8, 0.2, 0.6];
    let labels = [0_i64, 0, 1, 1, 0, 1];
    let scores_f32: Vec<f32> = scores_f64.iter().map(|score| *score as f32).collect();
    let auc = sk_roc_auc_score(&scores_f64, &labels, 1).expect("valid input");
    let auc_f32 = sk_roc_auc_score(&scores_f32, &labels, 1).expect("valid input");
    assert!((auc - auc_f32 as f64).abs() < 1e-5);

    let average_precision = sk_average_precision_score(&scores_f64, &labels, 1).expect("valid");
    let average_precision_f32 = sk_average_precision_score(&scores_f32, &labels, 1).expect("valid");
    assert!((average_precision - average_precision_f32 as f64).abs() < 1e-5);
}

/// f32 log loss matches f64 within tolerance.
#[test]
fn ranking_acceptance_f32_log_loss_matches_f64() {
    let labels = [0_i64, 1, 1];
    let probabilities = ndarray::array![[0.7_f64, 0.3], [0.4, 0.6], [0.2, 0.8]];
    let probabilities_f32 = probabilities.mapv(|value| value as f32);
    let loss = sk_log_loss(&labels, probabilities.view(), None).expect("valid input");
    let loss_f32 = sk_log_loss(&labels, probabilities_f32.view(), None).expect("valid input");
    assert!((loss - loss_f32 as f64).abs() < 1e-5);
}

/// Concurrent calls agree with the single-threaded result.
#[test]
fn ranking_acceptance_concurrent_calls_agree() {
    let (scores, labels) = large_scores_and_labels();
    let expected = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
    let observed: Vec<f64> = (0..8)
        .into_par_iter()
        .map(|_| {
            let auc = sk_roc_auc_score(&scores, &labels, 1).expect("valid input");
            let average_precision =
                sk_average_precision_score(&scores, &labels, 1).expect("valid input");
            assert!((average_precision - 1.0).abs() < 1e-12);
            auc
        })
        .collect();
    for auc in observed {
        assert_eq!(auc, expected);
    }
}

/// Concurrent multiclass calls agree with the sequential aggregate.
#[test]
fn ranking_acceptance_concurrent_multiclass_calls_agree() {
    let labels: Vec<i64> = (0..10_000).map(|index| (index % 3) as i64).collect();
    let mut scores = ndarray::Array2::<f64>::zeros((labels.len(), 3));
    for (row, label) in labels.iter().enumerate() {
        scores[[row, *label as usize]] = 1.0;
    }
    let expected = sk_multiclass_roc_auc_score(&scores.view(), &labels, SKRankingAveraging::Macro)
        .expect("valid input");
    let observed: Vec<f64> = (0..8)
        .into_par_iter()
        .map(|_| {
            sk_multiclass_roc_auc_score(&scores.view(), &labels, SKRankingAveraging::Macro)
                .expect("valid input")
        })
        .collect();
    for auc in observed {
        assert_eq!(auc, expected);
    }
}

/// Every public ranking type is `Send + Sync`.
#[test]
fn ranking_acceptance_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKAucAccumulator<f32>>();
    assert_send_sync::<SKAucAccumulator<f64>>();
    assert_send_sync::<SKRocCurve<f64>>();
    assert_send_sync::<SKPrecisionRecallCurve<f64>>();
    assert_send_sync::<SKAucScorer>();
    assert_send_sync::<SKAveragePrecisionScorer>();
    assert_send_sync::<SKLogLossScorer>();
    assert_send_sync::<SKBalancedAccuracyScorer>();
    assert_send_sync::<SKMatthewsCorrelationScorer>();
}

/// The scorer adapters are usable through a shared reference across threads.
#[test]
fn ranking_acceptance_scorer_is_shared_across_threads() {
    let scorer = SKAucScorer::new(1);
    let reference = &scorer;
    let values: Vec<bool> = (0..4)
        .into_par_iter()
        .map(|_| {
            let truth = ndarray::array![0_i64, 1];
            let view = sciencekit_common::SKTargetView::<f64>::Integer(truth.view());
            let probabilities = ndarray::array![[0.9_f64, 0.1], [0.2, 0.8]];
            use sciencekit_common::SKProbabilityScorer;
            let scored =
                <SKAucScorer as SKProbabilityScorer<f64, DummyModel>>::score_from_probabilities(
                    reference,
                    view,
                    probabilities.view(),
                )
                .expect("valid input");
            (scored - 1.0).abs() < 1e-12
        })
        .collect();
    assert!(values.into_iter().all(|value| value));
}

/// A model type only needed to name the scorer's generic parameter.
struct DummyModel;

impl<F: sciencekit_common::SKFloat> sciencekit_common::SKClassifierPredictor<F> for DummyModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<ndarray::Array1<i64>, SKError>
    where
        X: TryInto<sciencekit_common::SKDataView<'a, F>, Error = SKError>,
    {
        Ok(ndarray::Array1::zeros(0))
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<ndarray::Array2<F>, SKError>
    where
        X: TryInto<sciencekit_common::SKDataView<'a, F>, Error = SKError>,
    {
        Ok(ndarray::Array2::zeros((0, 0)))
    }
}
