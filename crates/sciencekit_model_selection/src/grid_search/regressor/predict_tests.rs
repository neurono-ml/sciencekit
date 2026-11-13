//! Regressor prediction delegation (spec `grid-search`, requirement
//! `Classifier and regressor prediction`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};

use super::SKGridSearchCVRegressor;
use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::split::kfold::SKKFold;

#[test]
fn grid_predict_regressor_responses_equal_the_best_model() {
    let features = Array2::zeros((10, 1));
    let targets = Array1::from_iter((0..10).map(|value| value as f64));
    let estimator = SKGridSearchCVRegressor::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(5.0),
            SKMockConstantRegressor::new(100.0),
        ])
        .splitter(SKKFold::build().number_of_splits(5).build().unwrap())
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();

    let expected = model
        .best_model()
        .unwrap()
        .predict(features.view())
        .unwrap();
    let predicted = model.predict(features.view()).unwrap();
    assert_eq!(predicted, expected);
    assert!(predicted.iter().all(|value| *value == 5.0));
}
