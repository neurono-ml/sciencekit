//! Companion tests for out-of-bag accuracy and R² estimation in generic
//! bagging models.

use ndarray::{Array1, Array2, Axis};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_metrics::regression::sk_r2_score;
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};

use crate::{SKBaggingClassifier, SKBaggingRegressor};

use super::sampling::sk_bagging_row_indices;
use crate::SKMaximumSamples;

#[test]
fn classifier_oob_accuracy_excludes_in_bag_rows_on_memorization_trap() {
    // Unique row IDs with alternating labels: full trees memorize in-bag rows
    // (~100% training accuracy), while honest OOB accuracy collapses because
    // adjacent IDs carry the opposite label.
    let rows = 40;
    let features = Array2::from_shape_fn((rows, 1), |(row, _)| row as f64);
    let labels = Array1::from_shape_fn(rows, |row| (row % 2) as i64);

    let base_tree = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    let model = SKBaggingClassifier::build(base_tree)
        .number_of_estimators(25)
        .oob_score(true)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let in_bag_predictions = model.predict_labels(features.view()).unwrap();
    let in_bag_correct = in_bag_predictions
        .iter()
        .zip(labels.iter())
        .filter(|(left, right)| left == right)
        .count();
    let in_bag_accuracy = (in_bag_correct as f64) / (rows as f64);
    let oob_accuracy = model.oob_score().unwrap();

    assert!(model.oob_scored_rows() > 30);
    assert!(in_bag_accuracy > 0.90);
    assert!(oob_accuracy < 0.45);
}

#[test]
fn regressor_oob_r2_matches_manual_routed_recomputation() {
    let rows = 36;
    let features = Array2::from_shape_fn((rows, 3), |(row, column)| {
        ((row * 2 + column * 3) as f64) * 0.15 + ((column + 1) as f64)
    });
    let targets = Array1::from_shape_fn(rows, |row| {
        2.5 * features[[row, 0]] - 1.5 * features[[row, 1]] + 0.75 * features[[row, 2]] + 1.0
    });

    let linear = SKLinearRegression::build().build().unwrap();
    let model = SKBaggingRegressor::build(linear)
        .number_of_estimators(16)
        .maximum_features(SKMaximumFeatures::Count(2))
        .oob_score(true)
        .random_seed(31)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();

    let mut sums = vec![0.0_f64; rows];
    let mut counts = vec![0usize; rows];
    (0..model.number_of_estimators()).for_each(|estimator_index| {
        let oob_rows = model.out_of_bag_rows(estimator_index).unwrap();
        let subset = model.estimator_features(estimator_index).unwrap();
        let base_model = model.estimator(estimator_index).unwrap();
        if !oob_rows.is_empty() {
            let gathered = features.select(Axis(0), oob_rows).select(Axis(1), subset);
            let predictions = base_model.predict(gathered.view()).unwrap();
            oob_rows
                .iter()
                .zip(predictions.iter())
                .for_each(|(&row, &prediction)| {
                    sums[row] += prediction;
                    counts[row] += 1;
                });
        }
    });

    let scored_pairs: Vec<(f64, f64)> = counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(row, &count)| (targets[row], sums[row] / (count as f64)))
        .collect();
    let true_oob = Array1::from_iter(scored_pairs.iter().map(|&(truth, _)| truth));
    let pred_oob = Array1::from_iter(scored_pairs.iter().map(|&(_, pred)| pred));
    let expected_r2 = sk_r2_score(true_oob.view(), pred_oob.view()).unwrap();

    assert_eq!(model.oob_scored_rows(), scored_pairs.len());
    assert!((model.oob_score().unwrap() - expected_r2).abs() < 1e-12);
}

#[test]
fn disabled_and_no_scored_rows_oob_report_unsupported_operation() {
    let features = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let targets = Array1::from_vec(vec![1.0_f64, 5.0]);
    let base_tree = SKDecisionTreeRegressor::<f64>::build().build().unwrap();

    let disabled = SKBaggingRegressor::build(base_tree.clone())
        .number_of_estimators(4)
        .oob_score(false)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(disabled.oob_scored_rows(), 0);
    assert!(disabled.out_of_bag_rows(0).is_none());
    assert!(matches!(
        disabled.oob_score().unwrap_err(),
        SKError::UnsupportedOperation {
            operation: "oob_score",
            ..
        }
    ));

    let full_coverage_seed = (0..10_000_u64)
        .find(|&seed| {
            let drawn = sk_bagging_row_indices(2, &SKMaximumSamples::All, true, seed, 0);
            drawn.contains(&0) && drawn.contains(&1)
        })
        .expect("a seed draws both [0, 1] on 2 rows");

    let empty_oob = SKBaggingRegressor::build(base_tree)
        .number_of_estimators(1)
        .oob_score(true)
        .random_seed(full_coverage_seed)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(empty_oob.oob_scored_rows(), 0);
    assert!(matches!(
        empty_oob.oob_score().unwrap_err(),
        SKError::UnsupportedOperation {
            operation: "oob_score",
            ..
        }
    ));
}
