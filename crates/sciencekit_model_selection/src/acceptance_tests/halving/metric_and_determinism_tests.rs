//! Scoring policies, determinism across seeds and thread pools, error-score
//! recovery and end-to-end `sciencekit_metrics` production.

use ndarray::{Array1, Array2};
use rayon::ThreadPoolBuilder;
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKRegressorPredictor, SKSupervisedFit,
    SKTargetView,
};
use sciencekit_metrics::classification::{SKAveraging, sk_accuracy_score, sk_f1_score};
use sciencekit_metrics::regression::{sk_mean_squared_error, sk_r2_score};

use super::super::fixtures::{cyclic_labels, dense_design};
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{SKClassifierScoring, SKErrorScore, SKRegressorScoring};
use crate::halving::{SKHalvingGridSearchClassifier, SKHalvingGridSearchRegressor};
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;
use crate::{SKTargetPartition, SKTrainTestSplitter, sk_train_test_split};

#[test]
fn halving_acceptance_every_classifier_and_regressor_scoring_policy_produces_finite_traces() {
    let features = dense_design(60, 3);
    let labels = cyclic_labels(60, 2);
    let clf_policies = [
        SKClassifierScoring::Accuracy,
        SKClassifierScoring::BalancedAccuracy,
        SKClassifierScoring::F1(SKAveraging::Macro),
        SKClassifierScoring::Precision(SKAveraging::Macro),
        SKClassifierScoring::Recall(SKAveraging::Macro),
        SKClassifierScoring::Auc { positive_label: 1 },
        SKClassifierScoring::LogLoss,
    ];
    for policy in clf_policies {
        let model = SKHalvingGridSearchClassifier::build()
            .candidates(vec![
                SKMockThresholdClassifier::new(10.0),
                SKMockThresholdClassifier::new(30.0),
                SKMockThresholdClassifier::new(50.0),
            ])
            .splitter(
                SKStratifiedKFold::build()
                    .number_of_splits(3)
                    .build()
                    .unwrap(),
            )
            .scoring(policy)
            .factor(3)
            .minimum_resource(18)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap();
        assert!(
            model.best_score().is_finite(),
            "{policy:?} produced a non-finite best score"
        );
    }

    let targets = Array1::from_iter((0..60).map(|row| (row % 9) as f64));
    for policy in [
        SKRegressorScoring::NegativeMeanSquaredError,
        SKRegressorScoring::NegativeMeanAbsoluteError,
        SKRegressorScoring::R2,
    ] {
        let model = SKHalvingGridSearchRegressor::build()
            .candidates(vec![
                SKMockConstantRegressor::new(1.0),
                SKMockConstantRegressor::new(4.0),
                SKMockConstantRegressor::new(7.0),
            ])
            .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
            .scoring(policy)
            .factor(3)
            .minimum_resource(18)
            .build()
            .unwrap()
            .fit(features.view(), targets.view())
            .unwrap();
        assert!(
            model.best_score().is_finite(),
            "{policy:?} produced a non-finite best score"
        );
    }
}

#[test]
fn halving_acceptance_determinism_error_score_and_held_out_metrics_compose() {
    let clf_features = Array2::from_shape_fn((120, 2), |(row, col)| (row + col) as f64);
    let clf_labels = Array1::from_iter((0..120).map(|row| i64::from(row >= 60)));
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.25)
        .shuffle(true)
        .stratify(true)
        .random_seed(13)
        .build()
        .unwrap();
    let split = sk_train_test_split(
        &splitter,
        SKDataView::Dense(clf_features.view()),
        Some(SKTargetView::Integer(clf_labels.view())),
    )
    .unwrap();
    let Some(SKTargetPartition::Integer(train_labels)) = split.train_targets() else {
        panic!("expected integer train targets");
    };
    let Some(SKTargetPartition::Integer(test_labels)) = split.test_targets() else {
        panic!("expected integer test targets");
    };

    let clf_candidates: Vec<SKMockThresholdClassifier> = (0..9)
        .map(|index| SKMockThresholdClassifier::new(((index + 1) * 12) as f64))
        .collect();
    let clf_search = SKHalvingGridSearchClassifier::build()
        .candidates(clf_candidates)
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(3)
                .shuffle(true)
                .random_seed(4)
                .build()
                .unwrap(),
        )
        .factor(3)
        .minimum_resource(18)
        .random_seed(29)
        .build()
        .unwrap();

    let run_in_pool = |threads: usize| {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                clf_search
                    .fit(split.train_features(), train_labels.view())
                    .unwrap()
            })
    };
    let single_thread = run_in_pool(1);
    let multi_thread = run_in_pool(4);
    assert_eq!(
        single_thread.elimination_trace(),
        multi_thread.elimination_trace()
    );

    let predicted_labels = single_thread.predict_labels(split.test_features()).unwrap();
    let truth_slice = test_labels.as_slice().unwrap();
    let pred_slice = predicted_labels.as_slice().unwrap();
    let accuracy: f64 = sk_accuracy_score(truth_slice, pred_slice).unwrap();
    let f1: f64 = sk_f1_score(truth_slice, pred_slice, None, SKAveraging::Macro).unwrap();
    assert!(accuracy > 0.90);
    assert!(f1 > 0.90);

    // Regression held-out evaluation + ErrorScore::Value(NaN) recovery.
    let reg_targets = Array1::from_iter((0..120).map(|row| if row < 60 { 9.0 } else { 11.0 }));
    let reg_model = SKHalvingGridSearchRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0).failing_when_target_present(11.0),
            SKMockConstantRegressor::new(5.0),
            SKMockConstantRegressor::new(10.0),
        ])
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .factor(3)
        .minimum_resource(18)
        .error_score(SKErrorScore::value(f64::NAN))
        .build()
        .unwrap()
        .fit(clf_features.view(), reg_targets.view())
        .unwrap();
    assert_eq!(reg_model.best_candidate_index(), 2);
    let reg_predictions = reg_model.predict(clf_features.view()).unwrap();
    let mse = sk_mean_squared_error(reg_targets.view(), reg_predictions.view()).unwrap();
    let r2 = sk_r2_score(reg_targets.view(), reg_predictions.view()).unwrap();
    assert!((mse - 1.0).abs() < 1e-12);
    assert!(r2.abs() < 1e-12);
}
