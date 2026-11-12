//! TDD tests for regressor halving refit and prediction delegation
//! (spec `halving-grid-search`, requirement `Refit and prediction`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};

use crate::grid_search::SKRegressorScoring;
use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::halving::SKHalvingGridSearchRegressor;
use crate::split::kfold::SKKFold;

#[test]
fn halving_refit_regressor_trains_winner_on_all_rows_and_delegates_predict() {
    // 27 candidates -> 9 -> 3 -> 1 across resources 10, 30, 90 on 100 rows!
    let features = Array2::from_shape_fn((100, 2), |(row, col)| (row * 2 + col) as f64);
    let targets = Array1::from_elem(100, 19.0);
    let candidates: Vec<SKMockConstantRegressor> = (0..27)
        .map(|index| SKMockConstantRegressor::new(index as f64))
        .collect();

    let estimator = SKHalvingGridSearchRegressor::build()
        .candidates(candidates)
        .splitter(SKKFold::build().number_of_splits(5).build().unwrap())
        .factor(3)
        .minimum_resource(10)
        .random_seed(9)
        .build()
        .unwrap();

    let model = estimator.fit(features.view(), targets.view()).unwrap();
    assert!(model.is_refitted());
    assert_eq!(model.number_of_rounds(), 3);
    assert_eq!(model.resource_schedule(), vec![10, 30, 90]);
    assert_eq!(model.candidate_count_trajectory(), vec![27, 9, 3, 1]);
    assert_eq!(model.best_candidate_index(), 19);
    assert_eq!(model.best_score(), 0.0);
    assert_eq!(
        model.scoring(),
        SKRegressorScoring::NegativeMeanSquaredError
    );
    // Refitted on all 100 rows, not just the 90-row final round subsample.
    assert_eq!(model.best_model().unwrap().training_rows(), 100);

    let predictions = model.predict(features.view()).unwrap();
    assert_eq!(predictions, Array1::from_elem(100, 19.0));
}
