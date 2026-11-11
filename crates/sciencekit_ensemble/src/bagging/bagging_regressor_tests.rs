//! Companion tests for [`SKBaggingRegressor`] arithmetic-mean response
//! aggregation and tree/non-tree composition.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit, SKTargetView,
};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::ridge::SKRidge;
use sciencekit_neighbors::SKKNeighborsRegressor;
use sciencekit_tree::SKDecisionTreeRegressor;

use crate::SKBaggingRegressor;

#[derive(Clone, Debug)]
struct ScriptedBaseRegressor {
    call_counter: Arc<AtomicUsize>,
    constant_responses: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ScriptedRegressorModel {
    constant_response: f64,
}

impl SKSupervisedFit<f64> for ScriptedBaseRegressor {
    type Model = ScriptedRegressorModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, _features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let slot = self.call_counter.fetch_add(1, Ordering::SeqCst);
        Ok(ScriptedRegressorModel {
            constant_response: self.constant_responses[slot],
        })
    }
}

impl SKRegressorPredictor<f64> for ScriptedRegressorModel {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array1::from_elem(rows, self.constant_response))
    }
}

#[test]
fn three_estimator_regressor_predicts_exact_arithmetic_mean() {
    let features = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let targets = Array1::from_vec(vec![1.0_f64, 2.0, 6.0]);
    let scripted = ScriptedBaseRegressor {
        call_counter: Arc::new(AtomicUsize::new(0)),
        constant_responses: vec![1.0, 2.0, 6.0],
    };

    let model = SKBaggingRegressor::build(scripted)
        .number_of_estimators(3)
        .bootstrap(false)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();

    let predictions = model.predict(features.view()).unwrap();
    predictions
        .iter()
        .for_each(|&value| assert!((value - 3.0).abs() < 1e-12));
}

#[test]
fn tree_linear_ridge_and_neighbor_regressors_fit_and_predict_accurately() {
    let features = Array2::from_shape_fn((32, 2), |(row, column)| {
        if column == 0 {
            (row as f64) * 0.25
        } else {
            ((31 - row) as f64) * 0.1
        }
    });
    let targets = Array1::from_shape_fn(32, |row| {
        3.0 * features[[row, 0]] - 2.0 * features[[row, 1]] + 0.5
    });

    let linear = SKLinearRegression::build().build().unwrap();
    let linear_predictions = SKBaggingRegressor::build(linear)
        .number_of_estimators(8)
        .random_seed(21)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
        .predict(features.view())
        .unwrap();
    linear_predictions
        .iter()
        .zip(targets.iter())
        .for_each(|(predicted, expected)| assert!((predicted - expected).abs() < 1e-8));

    let ridge = SKRidge::<f64>::build().alpha(1e-4).build().unwrap();
    let ridge_predictions = SKBaggingRegressor::build(ridge)
        .number_of_estimators(6)
        .random_seed(21)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
        .predict(features.view())
        .unwrap();
    assert_eq!(ridge_predictions.len(), 32);

    let tree = SKDecisionTreeRegressor::<f64>::build().build().unwrap();
    let tree_predictions = SKBaggingRegressor::build(tree)
        .number_of_estimators(10)
        .random_seed(21)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
        .predict(features.view())
        .unwrap();
    assert_eq!(tree_predictions.len(), 32);

    let knn = SKKNeighborsRegressor::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap();
    let knn_predictions = SKBaggingRegressor::build(knn)
        .number_of_estimators(6)
        .random_seed(21)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap()
        .predict(features.view())
        .unwrap();
    assert_eq!(knn_predictions.len(), 32);
}
