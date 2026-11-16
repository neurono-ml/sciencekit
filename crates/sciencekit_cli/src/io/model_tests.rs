//! Unit tests for the model artifact adapter.

use ndarray::array;
use tempfile::tempdir;

use super::{SKModelFormat, sk_inspect, sk_load_state, sk_save_state};
use crate::cli::commands::estimator::sk_fit_regressor;
use sciencekit::SKBackendKind;

fn sk_fitted_state() -> sciencekit::SKModelState<f64> {
    let features = array![[0.0_f64], [1.0], [2.0], [3.0]];
    let targets = array![1.0_f64, 3.0, 5.0, 7.0];
    sk_fit_regressor(
        "ridge",
        1.0,
        SKBackendKind::Faer,
        features.view(),
        targets.view(),
    )
    .expect("the model fits")
}

#[test]
fn safetensors_and_json_round_trip_the_same_state() {
    let state = sk_fitted_state();
    let directory = tempdir().expect("a temporary directory");
    for (name, format) in [
        ("model.safetensors", SKModelFormat::Safetensors),
        ("model.json", SKModelFormat::Json),
    ] {
        let path = directory.path().join(name);
        sk_save_state(&path, &state, format).expect("the state is written");
        let loaded = sk_load_state(&path).expect("the state reads back");
        assert_eq!(loaded.metadata().estimator_name(), "SKRidge");
        assert_eq!(loaded.number_of_tensors(), state.number_of_tensors());
        let document = sk_inspect(&path).expect("the artifact inspects");
        assert_eq!(document["estimator"], "SKRidge");
    }
}
