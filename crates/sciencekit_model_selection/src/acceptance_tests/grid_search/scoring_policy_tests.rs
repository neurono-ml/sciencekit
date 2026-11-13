//! Every scoring policy produces a finite score through the grid search.

use ndarray::Array1;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::super::fixtures::{cyclic_labels, dense_design};
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{
    SKAveraging, SKClassifierScoring, SKGridSearchCVClassifier, SKGridSearchCVRegressor,
    SKRegressorScoring,
};
use crate::split::kfold::SKKFold;

fn classifier_scores(policy: SKClassifierScoring) -> Vec<f64> {
    let features = dense_design(40, 3);
    let labels = cyclic_labels(40, 2);
    let estimator = SKGridSearchCVClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(2.0),
            SKMockThresholdClassifier::new(6.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .scoring(policy)
        .build()
        .unwrap();
    estimator
        .fit(features.view(), labels.view())
        .unwrap()
        .fold_scores()
        .iter()
        .copied()
        .collect()
}

#[test]
fn grid_acceptance_every_classification_policy_produces_finite_scores() {
    let policies = [
        SKClassifierScoring::Accuracy,
        SKClassifierScoring::BalancedAccuracy,
        SKClassifierScoring::F1(SKAveraging::Macro),
        SKClassifierScoring::Precision(SKAveraging::Macro),
        SKClassifierScoring::Recall(SKAveraging::Macro),
        SKClassifierScoring::Auc { positive_label: 1 },
        SKClassifierScoring::LogLoss,
    ];
    for policy in policies {
        let scores = classifier_scores(policy);
        assert_eq!(scores.len(), 8);
        assert!(
            scores.iter().all(|score| score.is_finite()),
            "{policy:?} produced a non-finite score"
        );
    }
}

#[test]
fn grid_acceptance_every_regression_policy_produces_finite_scores() {
    let features = dense_design(40, 3);
    let targets = Array1::from_iter((0..40).map(|row| (row % 7) as f64));
    let policies = [
        SKRegressorScoring::NegativeMeanSquaredError,
        SKRegressorScoring::NegativeMeanAbsoluteError,
        SKRegressorScoring::R2,
    ];
    for policy in policies {
        let estimator = SKGridSearchCVRegressor::build()
            .candidates(vec![
                SKMockConstantRegressor::new(0.0),
                SKMockConstantRegressor::new(3.0),
            ])
            .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
            .scoring(policy)
            .build()
            .unwrap();
        let model = estimator.fit(features.view(), targets.view()).unwrap();
        assert!(
            model.fold_scores().iter().all(|score| score.is_finite()),
            "{policy:?} produced a non-finite score"
        );
    }
}
