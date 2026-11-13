//! Companion tests for `SKClustererPredictor` on `SKDBSCANModel`
//! (spec `dbscan`, decision 4).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKClusterLabel, SKClustererPredictor, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKDBSCAN, SKDBSCANAlgorithm};

fn training_with_border_and_noise() -> Array2<f64> {
    array![
        [0.0_f64, 0.0], // 0: core Cluster(0)
        [0.4, 0.0],     // 1: core Cluster(0)
        [0.0, 0.4],     // 2: core Cluster(0)
        [-0.85, 0.0],   // 3: border of Cluster(0) when eps = 0.9, minimum_samples = 3
        [8.0, 8.0],     // 4: core Cluster(1)
        [8.4, 8.0],     // 5: core Cluster(1)
        [8.0, 8.4],     // 6: core Cluster(1)
        [50.0, 50.0],   // 7: isolated noise
    ]
}

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

#[test]
fn dbscan_predict_assigns_nearby_queries_to_nearest_core_cluster_and_distant_to_noise() {
    let model = SKDBSCAN::<f64>::build()
        .eps(0.9)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(&training_with_border_and_noise())
        .unwrap();

    let queries = array![
        [0.1_f64, 0.1], // within eps of Cluster(0) core samples -> Cluster(0)
        [8.1, 8.1],     // within eps of Cluster(1) core samples -> Cluster(1)
        [-0.9, 0.0],    // exactly at boundary distance 0.9 from core row 0 -> Cluster(0)
        [-1.3, 0.0],    // 0.45 from border row 3, 1.3 > 0.9 from nearest core row 0 -> Noise
        [25.0, 25.0],   // far from every core sample -> Noise
    ];

    let assignments = model.predict_clusters(&queries).unwrap();
    assert_eq!(
        assignments.to_vec(),
        vec![
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(1),
            SKClusterLabel::cluster(0),
            SKClusterLabel::Noise,
            SKClusterLabel::Noise,
        ]
    );
}

#[test]
fn dbscan_predict_returns_noise_when_model_found_no_core_samples() {
    let sparse_points = array![[0.0_f64, 0.0], [5.0, 5.0], [10.0, 10.0]];
    let model = SKDBSCAN::<f64>::build()
        .eps(0.5)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(&sparse_points)
        .unwrap();
    assert_eq!(model.number_of_clusters(), 0);
    assert!(model.core_sample_indices().is_empty());

    let queries = array![[0.0_f64, 0.0], [5.0, 5.0]];
    let assignments = model.predict_clusters(&queries).unwrap();
    assert_eq!(
        assignments.to_vec(),
        vec![SKClusterLabel::Noise, SKClusterLabel::Noise]
    );
}

#[test]
fn dbscan_predict_covers_all_four_dense_and_sparse_combinations_without_densifying() {
    let dense_train = training_with_border_and_noise();
    let sparse_train = csr_from_dense(&dense_train);
    let dense_queries = array![[0.1_f64, 0.0], [8.1, 8.1], [-1.3, 0.0], [30.0, 0.0],];
    let sparse_queries = csr_from_dense(&dense_queries);

    let dense_model = SKDBSCAN::<f64>::build()
        .eps(0.9)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(&dense_train)
        .unwrap();
    let sparse_model = SKDBSCAN::<f64>::build()
        .eps(0.9)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(sparse_train.view())
        .unwrap();

    let dense_on_dense = dense_model.predict_clusters(&dense_queries).unwrap();
    let dense_on_sparse = dense_model.predict_clusters(sparse_queries.view()).unwrap();
    let sparse_on_dense = sparse_model.predict_clusters(&dense_queries).unwrap();
    let sparse_on_sparse = sparse_model
        .predict_clusters(sparse_queries.view())
        .unwrap();

    assert_eq!(dense_on_dense, dense_on_sparse);
    assert_eq!(dense_on_dense, sparse_on_dense);
    assert_eq!(dense_on_dense, sparse_on_sparse);
}

#[test]
fn dbscan_predict_breaks_equidistant_core_ties_by_smallest_training_index() {
    let data = array![
        [0.0_f64, 0.0],
        [-0.2, 0.0],
        [-0.2, 0.2],
        [2.0, 0.0],
        [2.2, 0.0],
        [2.2, 0.2],
    ];
    // Query [1.0, 0.0] is distance 1.0 from core row 0 (Cluster 0) and 1.0 from core row 3 (Cluster 1).
    let query = array![[1.0_f64, 0.0]];

    for algorithm in [
        SKDBSCANAlgorithm::KdTree,
        SKDBSCANAlgorithm::BallTree,
        SKDBSCANAlgorithm::BruteForce,
    ] {
        let model = SKDBSCAN::<f64>::build()
            .eps(1.0)
            .minimum_samples(3)
            .algorithm(algorithm)
            .build()
            .unwrap()
            .fit(&data)
            .unwrap();
        let predicted = model.predict_clusters(&query).unwrap();
        assert_eq!(predicted.to_vec(), vec![SKClusterLabel::cluster(0)]);
    }
}
