//! Companion tests: the isolation forest round-trips its tree ensemble.

use ndarray::array;
use sciencekit_common::{SKBackendKind, sk_assert_export_round_trip};

use super::SKContamination;
use super::SKIsolationForest;

#[test]
fn isolation_forest_round_trips_trees_offset_and_scores() {
    let training = array![
        [0.0_f64, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
        [0.0, 0.2],
    ];
    let model = SKIsolationForest::<f64>::build()
        .number_of_trees(8)
        .random_seed(5)
        .contamination(SKContamination::Fraction(0.2))
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_trees(), model.number_of_trees());
    assert_eq!(imported.offset(), model.offset());
    assert_eq!(
        model.score_samples(training.view()).unwrap(),
        imported.score_samples(training.view()).unwrap()
    );
    assert_eq!(
        model.predict(training.view()).unwrap(),
        imported.predict(training.view()).unwrap()
    );
}
