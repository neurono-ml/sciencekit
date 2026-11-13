//! Metric-production acceptance tests through `sciencekit_metrics`
//! (PRD §8.7, spec `support-vector-machines`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKLabelScorer, SKProbabilityScorer,
    SKRegressorPredictor, SKSupervisedFit, SKSupervisedScorer, SKTargetView,
};
use sciencekit_math::SKKernel;
use sciencekit_metrics::classification::{SKAveraging, sk_accuracy_score, sk_f1_score};
use sciencekit_metrics::regression::sk_r2_score;
use sciencekit_metrics::scorers::{SKAccuracyScorer, SKF1Scorer, SKLogLossScorer, SKR2Scorer};

use crate::{SKSVC, SKSVR};

#[test]
fn acceptance_metrics_evaluate_svc_labels_probabilities_and_svr_responses() {
    let rows = 90usize;
    let features = Array2::from_shape_fn((rows, 2), |(r, c)| {
        let class = r / 30;
        let center = match class {
            0 => [-3.0, -2.0],
            1 => [0.0, 3.0],
            _ => [3.0, -2.0],
        };
        center[c] + ((r * 7 + c * 5) % 11) as f64 * 0.06
    });
    let labels = Array1::from_shape_fn(rows, |r| (r / 30) as i64);

    let svc = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(2.0)
        .probability(true)
        .random_seed(21)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let predicted = svc.predict_labels(features.view()).unwrap();
    let truth = labels.to_vec();
    let pred_vec = predicted.to_vec();

    let accuracy: f64 = sk_accuracy_score(&truth, &pred_vec).unwrap();
    let macro_f1: f64 = sk_f1_score(&truth, &pred_vec, None, SKAveraging::Macro).unwrap();
    assert!(accuracy > 0.98);
    assert!(macro_f1 > 0.98);

    let x_view = SKDataView::Dense(features.view());
    let y_view = SKTargetView::Integer(labels.view());
    let scorer_acc = SKAccuracyScorer.score(&svc, x_view, y_view).unwrap();
    let scorer_f1 = SKF1Scorer::new(SKAveraging::Macro)
        .score(&svc, x_view, y_view)
        .unwrap();
    let log_loss_scorer = SKLogLossScorer::default();
    let neg_log_loss = log_loss_scorer.score(&svc, x_view, y_view).unwrap();
    assert!((scorer_acc - accuracy).abs() < 1e-12);
    assert!((scorer_f1 - macro_f1).abs() < 1e-12);
    assert!(neg_log_loss.is_finite() && neg_log_loss > -0.5);

    let reg_x = Array2::from_shape_fn((80, 1), |(r, _)| -2.5 + 5.0 * (r as f64) / 79.0);
    let reg_y = Array1::from_shape_fn(80, |r| reg_x[[r, 0]].sin() + 0.2 * reg_x[[r, 0]]);
    let svr = SKSVR::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(10.0)
        .epsilon(0.03)
        .build()
        .unwrap()
        .fit(reg_x.view(), reg_y.view())
        .unwrap();

    let reg_pred = svr.predict(reg_x.view()).unwrap();
    let r2 = sk_r2_score(reg_y.view(), reg_pred.view()).unwrap();
    let scorer_r2 = SKR2Scorer
        .score(
            &svr,
            SKDataView::Dense(reg_x.view()),
            SKTargetView::Continuous(reg_y.view()),
        )
        .unwrap();
    assert!(r2 > 0.98);
    assert!((r2 - scorer_r2).abs() < 1e-12);
}
