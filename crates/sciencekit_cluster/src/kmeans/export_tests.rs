//! Companion tests: KMeans and DBSCAN round-trip their learned state.

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClustererPredictor, SKUnsupervisedFit, sk_assert_export_round_trip,
};

use crate::dbscan::SKDBSCAN;
use crate::kmeans::SKKMeans;

#[test]
fn kmeans_round_trips_centers_labels_and_backend() {
    let features = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1],
    ];
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(2)
        .random_seed(17)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_clusters(), 2);
    assert_eq!(imported.cluster_centers(), model.cluster_centers());
    assert_eq!(
        model.predict_clusters(features.view()).unwrap(),
        imported.predict_clusters(features.view()).unwrap()
    );
}

#[test]
fn dbscan_round_trips_labels_and_nearest_core_assignment() {
    let training = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1],
    ];
    let model = SKDBSCAN::<f64>::build()
        .eps(0.9)
        .minimum_samples(3)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(&training)
        .unwrap();
    let imported = sk_assert_export_round_trip::<f64, _>(&model).unwrap();
    assert_eq!(imported.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(imported.number_of_clusters(), model.number_of_clusters());
    assert_eq!(imported.core_sample_indices(), model.core_sample_indices());
    let queries = array![[0.05_f64, 0.05], [10.05, 10.05], [25.0, 25.0]];
    assert_eq!(
        model.predict_clusters(queries.view()).unwrap(),
        imported.predict_clusters(queries.view()).unwrap()
    );
}
