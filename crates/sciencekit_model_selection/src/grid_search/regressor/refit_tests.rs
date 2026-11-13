//! Refit and best-model accessors for the regressor grid search
//! (spec `grid-search`, requirement `Refit`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::SKGridSearchCVRegressor;
use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::grid_search::scoring::SKRegressorScoring;
use crate::split::kfold::SKKFold;

fn fixtures() -> (Array2<f64>, Array1<f64>) {
    let features = Array2::zeros((10, 1));
    let targets = Array1::from_iter((0..10).map(|value| value as f64));
    (features, targets)
}

fn estimator(refit: bool) -> SKGridSearchCVRegressor<f64, SKMockConstantRegressor> {
    SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(5.0),
            SKMockConstantRegressor::new(100.0),
        ])
        .splitter(SKKFold::build().number_of_splits(5).build().unwrap())
        .scoring(SKRegressorScoring::NegativeMeanSquaredError)
        .refit(refit)
        .build()
        .unwrap()
}

#[test]
fn grid_refit_regressor_retrains_the_winner_on_every_row() {
    let (features, targets) = fixtures();
    let model = estimator(true)
        .fit(features.view(), targets.view())
        .unwrap();

    assert!(model.is_refitted());
    assert_eq!(model.best_candidate_index(), 1);
    assert_eq!(model.best_model().unwrap().training_rows(), 10);
    assert_eq!(model.mean_scores().len(), 3);
    assert_eq!(model.standard_deviations().len(), 3);
}

#[test]
fn grid_refit_regressor_off_keeps_only_the_results() {
    let (features, targets) = fixtures();
    let model = estimator(false)
        .fit(features.view(), targets.view())
        .unwrap();

    assert!(!model.is_refitted());
    assert!(model.best_model().is_none());
    assert_eq!(model.ranked_candidate_indices(), vec![1, 0, 2]);
}
