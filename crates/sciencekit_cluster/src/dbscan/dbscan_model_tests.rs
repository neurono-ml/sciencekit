//! Companion tests for `SKUnsupervisedFit`, `fit_predict` and `SKDBSCANModel`
//! accessors (spec `dbscan`, decision 6).

use std::collections::HashSet;

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClusterLabel, SKExecutionMode, SKUnsupervisedFit,
};

use super::{SKDBSCAN, SKDBSCANMetric, SKDBSCANModel, SKDBSCANSearchStrategy};

fn assert_send_and_sync<T: Send + Sync>() {}

fn two_clusters_and_noise() -> Array2<f64> {
    array![
        [0.0_f64, 0.0],
        [0.4, 0.0],
        [0.0, 0.4],
        [-0.85, 0.0], // border point of cluster 0 when eps = 0.9, minimum_samples = 3
        [8.0, 8.0],
        [8.4, 8.0],
        [8.0, 8.4],
        [50.0, 50.0], // isolated noise
    ]
}

#[test]
fn dbscan_model_accessors_describe_the_fitted_partition() {
    assert_send_and_sync::<SKDBSCANModel<f64>>();
    assert_send_and_sync::<SKDBSCANModel<f32>>();

    let data = two_clusters_and_noise();
    let estimator = SKDBSCAN::<f64>::build()
        .eps(0.9)
        .minimum_samples(3)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    let model = estimator.fit(&data).unwrap();

    assert_eq!(model.labels().len(), data.nrows());
    assert_eq!(model.number_of_features(), 2);
    assert_eq!(model.eps(), 0.9);
    assert_eq!(model.minimum_samples(), 3);
    assert_eq!(model.metric(), SKDBSCANMetric::Euclidean);
    assert_eq!(model.search_strategy(), SKDBSCANSearchStrategy::KdTree);
    assert_eq!(model.training_representation(), "dense");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );

    let distinct_clusters: HashSet<u32> = model
        .labels()
        .iter()
        .filter_map(|label| label.cluster_index())
        .collect();
    assert_eq!(model.number_of_clusters(), 2);
    assert_eq!(distinct_clusters.len(), model.number_of_clusters());
    assert_eq!(model.core_sample_indices().to_vec(), vec![0, 1, 2, 4, 5, 6]);
    for &core_index in model.core_sample_indices().iter() {
        assert!(core_index < data.nrows());
        assert!(!model.labels()[core_index].is_noise());
    }
    assert_eq!(model.labels()[3], SKClusterLabel::cluster(0));
    assert_eq!(model.labels()[7], SKClusterLabel::Noise);
}

#[test]
fn dbscan_model_fit_predict_matches_fit_labels_and_supports_f32() {
    let data64 = two_clusters_and_noise();
    let data32 = data64.mapv(|value| value as f32);

    let estimator64 = SKDBSCAN::<f64>::build()
        .eps(0.9)
        .minimum_samples(3)
        .build()
        .unwrap();
    let model64 = estimator64.fit(&data64).unwrap();
    let direct64 = estimator64.fit_predict(&data64).unwrap();
    assert_eq!(direct64, model64.labels());

    let model32 = SKDBSCAN::<f32>::build()
        .eps(0.9_f32)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(&data32)
        .unwrap();
    assert_eq!(model32.labels(), model64.labels());
    assert_eq!(model32.core_sample_indices(), model64.core_sample_indices());
    assert_eq!(model32.number_of_clusters(), model64.number_of_clusters());
}
