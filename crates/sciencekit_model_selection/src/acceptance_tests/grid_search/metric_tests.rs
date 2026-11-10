//! Determinism, metric aggregation and failure/refit acceptance.

use ndarray::{Array1, Array2, array};
use rayon::ThreadPoolBuilder;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::super::fixtures::dense_design;
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{SKErrorScore, SKGridSearchCVClassifier, SKGridSearchCVRegressor};
use crate::split::kfold::SKKFold;

fn regression_setup() -> (Array2<f64>, Array1<f64>) {
    let features = dense_design(40, 3);
    let targets = Array1::from_iter((0..40).map(|row| (row % 9) as f64));
    (features, targets)
}

#[test]
fn grid_acceptance_fold_scores_do_not_depend_on_thread_count() {
    let (features, targets) = regression_setup();
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(4.0),
            SKMockConstantRegressor::new(8.0),
        ])
        .splitter(
            SKKFold::build()
                .number_of_splits(4)
                .shuffle(true)
                .random_seed(11)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let run = |threads: usize| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            estimator
                .fit(features.view(), targets.view())
                .unwrap()
                .fold_scores()
                .to_owned()
        })
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn grid_acceptance_aggregated_scores_match_the_fold_matrix() {
    let (features, targets) = regression_setup();
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(4.0),
            SKMockConstantRegressor::new(8.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();

    for (candidate, row) in model.fold_scores().rows().into_iter().enumerate() {
        let expected = row.iter().sum::<f64>() / row.len() as f64;
        assert!((model.mean_scores()[candidate] - expected).abs() < 1e-12);
    }
    let best = model.best_candidate_index();
    assert_eq!(model.best_score(), model.mean_scores()[best]);
}

#[test]
fn grid_acceptance_error_score_and_refit_policies_compose() {
    let (features, targets) = regression_setup();
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0).failing_when_target_present(3.0),
            SKMockConstantRegressor::new(4.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .error_score(SKErrorScore::value(f64::NAN))
        .refit(false)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();

    assert!(!model.is_refitted());
    assert!(model.fold_scores().iter().any(|score| score.is_nan()));
    assert_eq!(model.best_candidate_index(), 1);
}

#[test]
fn grid_acceptance_classifier_and_regressor_produce_metrics() {
    let (features, targets) = regression_setup();
    let labels = Array1::from_iter((0..40).map(|row| i64::from(row % 2 == 0)));

    let classifier = SKGridSearchCVClassifier::build()
        .candidates(vec![
            SKMockThresholdClassifier::new(2.0),
            SKMockThresholdClassifier::new(6.0),
        ])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert!(classifier.best_score() >= 0.0 && classifier.best_score() <= 1.0);

    let regressor = SKGridSearchCVRegressor::build()
        .candidates(vec![SKMockConstantRegressor::new(4.0)])
        .splitter(SKKFold::build().number_of_splits(4).build().unwrap())
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert!(regressor.best_score() <= 0.0);
    assert_eq!(array![regressor.best_candidate_index()].len(), 1);
}
