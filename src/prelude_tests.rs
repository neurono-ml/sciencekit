//! Compile and run proof that the prelude is enough for a typical program.
//!
//! This is the executable form of the "Prelude compiles a basic program"
//! scenario: a program that imports only `sciencekit::prelude::*` builds a
//! scaler and a regressor, fits them and predicts.

use ndarray::array;

use crate::prelude::*;

#[test]
fn prelude_builds_fits_and_predicts() {
    let features = array![[0.0_f64, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
    let targets = array![1.0_f64, 2.0, 3.0, 4.0];

    let scaler = SKStandardScaler::build().build().unwrap();
    let scaled = scaler.fit(features.view()).unwrap();
    let scaled_features = scaled.transform(features.view()).unwrap();

    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(scaled_features.view(), targets.view())
        .unwrap();
    let predictions = model.predict(scaled_features.view()).unwrap();

    assert_eq!(predictions.len(), features.nrows());
    for (prediction, target) in predictions.iter().zip(targets.iter()) {
        assert!((prediction - target).abs() < 1e-6);
    }
}

#[test]
fn prelude_exposes_the_shared_contract_traits() {
    fn fit_contract<T: SKSupervisedFit<f64>>() {}
    fn unsupervised_contract<T: SKUnsupervisedFit<f64>>() {}
    fn export_contract<T: SKExportable<f64>>() {}

    let _ = fit_contract::<SKLinearRegression>;
    let _ = unsupervised_contract::<SKStandardScaler>;
    let _ = export_contract::<SKStandardScalerModel<f64>>;
}

#[test]
fn prelude_exposes_metrics_for_a_fitted_model() {
    let truth = [0_i64, 1, 1, 0];
    let predicted = [0_i64, 1, 0, 0];
    let score: f64 = sk_accuracy_score(&truth, &predicted).unwrap();
    assert!((score - 0.75).abs() < 1e-12);
}
