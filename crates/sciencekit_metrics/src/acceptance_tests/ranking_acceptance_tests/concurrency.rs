//! Concurrency and thread-safety acceptance of the ranking metrics.

use rayon::prelude::*;
use sciencekit_common::{SKProbabilityScorer, SKTargetView};

use crate::ranking::{
    SKAucAccumulator, SKPrecisionRecallCurve, SKRankingAveraging, SKRocCurve,
    sk_average_precision_score, sk_multiclass_roc_auc_score, sk_roc_auc_score,
};
use crate::scorers::{
    SKAucScorer, SKAveragePrecisionScorer, SKBalancedAccuracyScorer, SKLogLossScorer,
    SKMatthewsCorrelationScorer,
};

use super::fixtures::{DummyModel, large_scores_and_labels};

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
            let view = SKTargetView::<f64>::Integer(truth.view());
            let probabilities = ndarray::array![[0.9_f64, 0.1], [0.2, 0.8]];
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
