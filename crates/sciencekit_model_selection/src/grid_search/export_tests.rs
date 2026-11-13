//! Companion test: a grid-search export carries the winner and selection data.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKExportable, SKImportable, SKSupervisedFit};

use crate::grid_search::SKGridSearchCVRegressor;
use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::grid_search::scoring::SKRegressorScoring;
use crate::split::kfold::SKKFold;

#[test]
fn grid_search_export_carries_the_best_model_and_selection_metadata() {
    let features: Array2<f64> = Array2::zeros((10, 1));
    let targets = Array1::from_iter((0..10).map(|value| value as f64));
    let estimator = SKGridSearchCVRegressor::<f64, SKMockConstantRegressor>::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(5.0),
            SKMockConstantRegressor::new(100.0),
        ])
        .splitter(SKKFold::build().number_of_splits(5).build().unwrap())
        .scoring(SKRegressorScoring::NegativeMeanSquaredError)
        .refit(true)
        .build()
        .unwrap();
    let model = estimator.fit(features.view(), targets.view()).unwrap();
    let state = model.export_state().unwrap();
    assert_eq!(
        state
            .metadata()
            .parameter("grid_search.best_candidate_index")
            .and_then(sciencekit_common::SKParameterValue::as_integer),
        Some(1)
    );
    assert_eq!(
        state
            .metadata()
            .parameter("grid_search.candidate_count")
            .and_then(sciencekit_common::SKParameterValue::as_integer),
        Some(3)
    );
    assert!(state.tensor("constant").is_some());
}

#[test]
fn grid_search_import_returns_a_typed_unsupported_operation() {
    let features: Array2<f64> = Array2::zeros((6, 1));
    let targets = Array1::from_iter((0..6).map(|value| value as f64));
    let model = SKGridSearchCVRegressor::<f64, SKMockConstantRegressor>::build()
        .candidates(vec![
            SKMockConstantRegressor::new(0.0),
            SKMockConstantRegressor::new(1.0),
        ])
        .splitter(SKKFold::build().number_of_splits(2).build().unwrap())
        .scoring(SKRegressorScoring::NegativeMeanSquaredError)
        .refit(true)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = model.export_state().unwrap();
    assert!(matches!(
        crate::grid_search::SKGridSearchCVRegressorModel::<f64, SKMockConstantRegressor>::import_state(
            &state
        ),
        Err(sciencekit_common::SKError::UnsupportedOperation {
            operation: "import_state",
            ..
        })
    ));
}
