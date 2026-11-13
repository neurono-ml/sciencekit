//! Fitted fixtures shared by the native-import tests.

use ndarray::{Array2, array};
use sciencekit_cluster::kmeans::{SKKMeans, SKKMeansModel};
use sciencekit_common::{SKBuilder, SKSupervisedFit, SKUnsupervisedFit};
use sciencekit_ensemble::random_forest::{SKRandomForestClassifier, SKRandomForestClassifierModel};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegressionModel;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sciencekit_preprocessing::standard_scaler::model::SKStandardScalerModel;
use sciencekit_tree::{SKDecisionTreeRegressor, SKDecisionTreeRegressorModel};

pub(super) fn features() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 2.0],
    ]
}

fn targets() -> ndarray::Array1<f64> {
    array![1.0_f64, 3.0, 4.0, 14.0, 11.0, 16.0]
}

pub(super) fn linear_regression() -> SKLinearRegressionModel<f64> {
    SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features().view(), targets().view())
        .unwrap()
}

pub(super) fn logistic_regression() -> SKLogisticRegressionModel<f64> {
    let labels = ["low", "low", "low", "high", "high", "high"];
    SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features().view(), labels.as_slice())
        .unwrap()
}

pub(super) fn standard_scaler() -> SKStandardScalerModel<f64> {
    SKStandardScaler::build()
        .build()
        .unwrap()
        .fit(features().view())
        .unwrap()
}

pub(super) fn decision_tree_regressor() -> SKDecisionTreeRegressorModel<f64> {
    SKDecisionTreeRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(features().view(), targets().view())
        .unwrap()
}

pub(super) fn random_forest_classifier() -> SKRandomForestClassifierModel<f64> {
    let labels = ["a", "a", "a", "b", "b", "b"];
    SKRandomForestClassifier::<f64>::build()
        .number_of_trees(4)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(features().view(), labels.as_slice())
        .unwrap()
}

pub(super) fn kmeans() -> SKKMeansModel<f64> {
    SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(features().view())
        .unwrap()
}
