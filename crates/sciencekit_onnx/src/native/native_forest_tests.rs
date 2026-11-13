//! Companion tests: an exported random-forest regressor imports natively and
//! predicts the same values (spec `native-model-import`, requirement "ONNX
//! native import").

use ndarray::{Array1, Array2, array};

use super::SKOnnxNativeImporter;
use crate::export::SKToOnnx;

fn features() -> Array2<f32> {
    array![
        [0.0_f32, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 2.0],
        [5.0, 4.0],
        [1.5, 2.5],
    ]
}

fn targets() -> Array1<f32> {
    array![1.0_f32, 3.0, 4.0, 14.0, 11.0, 16.0, 21.0, 9.0]
}

fn probe() -> Array2<f32> {
    array![[0.5_f32, 0.5], [2.5, 2.0], [4.5, 3.0], [-1.0, -1.0]]
}

/// A forest regressor exported to ONNX rebuilds with the same mean prediction.
///
/// The ONNX-ML operator aggregates a *sum* while the native forest predicts a
/// *mean*, so import is obliged to scale each tree by the member count and
/// divide again; the two rounding orders differ by at most a few ulps, which is
/// why this check is tolerance-based rather than bit-exact.
#[test]
fn native_onnx_round_trips_a_random_forest_regressor() {
    use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};
    use sciencekit_ensemble::random_forest::SKRandomForestRegressor;

    let model = SKRandomForestRegressor::<f32>::build()
        .number_of_trees(4)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(features().view(), targets().view())
        .unwrap();
    let graph = model.to_onnx().unwrap();
    let imported = SKOnnxNativeImporter::builder()
        .model(graph)
        .build::<f32>()
        .unwrap();
    let expected = model.predict(probe().view()).unwrap();
    let actual = imported.predict_regression(probe().view()).unwrap();
    let difference = expected
        .iter()
        .zip(&actual)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    assert!(difference <= 1e-6, "max |difference| {difference:.3e}");
    assert_eq!(
        imported.descriptor().estimator_name(),
        "SKRandomForestRegressor"
    );
}
