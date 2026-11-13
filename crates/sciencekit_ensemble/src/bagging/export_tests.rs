//! Companion tests: bagging states are explicit about their import limitation.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKExportable, SKImportable, SKSupervisedFit};

use crate::bagging::{SKBaggingClassifierModel, SKBaggingRegressorModel};
use crate::{SKBaggingClassifier, SKBaggingRegressor};
use sciencekit_tree::{
    SKDecisionTreeClassifier, SKDecisionTreeClassifierModel, SKDecisionTreeRegressor,
    SKDecisionTreeRegressorModel,
};

#[test]
fn bagging_regressor_import_returns_a_typed_unsupported_operation() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0];
    let model = SKBaggingRegressor::build(SKDecisionTreeRegressor::<f64>::build().build().unwrap())
        .number_of_estimators(2)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    let state = model.export_state().unwrap();
    assert!(matches!(
        SKBaggingRegressorModel::<f64, SKDecisionTreeRegressorModel<f64>>::import_state(&state),
        Err(sciencekit_common::SKError::UnsupportedOperation {
            operation: "import_state",
            ..
        })
    ));
}

#[test]
fn bagging_classifier_import_returns_a_typed_unsupported_operation() {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let labels = array![0_i64, 0, 1, 1];
    let model =
        SKBaggingClassifier::build(SKDecisionTreeClassifier::<f64>::build().build().unwrap())
            .number_of_estimators(2)
            .random_seed(7)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap();
    let state = model.export_state().unwrap();
    assert!(matches!(
        SKBaggingClassifierModel::<f64, SKDecisionTreeClassifierModel<f64>>::import_state(&state),
        Err(sciencekit_common::SKError::UnsupportedOperation {
            operation: "import_state",
            ..
        })
    ));
}
