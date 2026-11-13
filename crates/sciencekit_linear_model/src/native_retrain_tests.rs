//! Companion tests: continued fitting from an imported linear-model state
//! (spec `native-model-import`, requirement "Continued training").
//!
//! A stochastic-gradient model keeps its learned iterate across a round trip, so
//! `partial_fit` must continue from the imported coefficients rather than start
//! again from zero.

use ndarray::array;
use sciencekit_common::{
    SKBuilder, SKExportable, SKImportable, SKRegressorPredictor, SKSupervisedFit,
};

use crate::sgd_regressor::SKSGDRegressor;
use crate::sgd_regressor::model::SKSGDRegressorModel;

fn features() -> ndarray::Array2<f64> {
    array![[0.0_f64, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]]
}

fn extra_features() -> ndarray::Array2<f64> {
    array![[4.0_f64, 3.0], [5.0, 2.0]]
}

#[test]
fn native_retrain_sgd_regressor_continues_from_the_imported_iterate() {
    let targets = array![1.0_f64, 2.0, 5.0, 4.0];
    let original = SKSGDRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(features().view(), targets.view())
        .unwrap();
    let state = original.export_state().unwrap();
    let mut imported = SKSGDRegressorModel::<f64>::import_state(&state).unwrap();
    let before = imported.coefficients().to_owned();
    let extra = array![7.0_f64, 8.0];
    imported
        .partial_fit(extra_features().view(), extra.view())
        .unwrap();
    let after = imported.coefficients().to_owned();

    assert_ne!(
        before, after,
        "continued fitting must update the coefficients"
    );
    assert!(before != after);
    assert!(
        imported
            .predict(features().view())
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );

    let cold = SKSGDRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(extra_features().view(), extra.view())
        .unwrap();
    assert_ne!(
        cold.coefficients().to_owned(),
        after,
        "the imported iterate must seed the update, not a cold restart"
    );
}
