//! Companion tests for DBSCAN core detection, expansion and deterministic
//! cluster identifiers (spec `dbscan`, decisions 2 and 3).

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKClusterLabel, SKDataView, SKExecutionMode};
use sciencekit_neighbors::SKTrainingData;

use super::config::SKDBSCANConfiguration;
use super::expansion::{SKDBSCANExpansion, sk_run_dbscan_partition};
use super::{SKDBSCANAlgorithm, SKDBSCANMetric};

fn expand_dense(data: &Array2<f64>, eps: f64, minimum_samples: usize) -> SKDBSCANExpansion {
    let configuration = SKDBSCANConfiguration {
        eps,
        minimum_samples,
        metric: SKDBSCANMetric::Euclidean,
        algorithm: SKDBSCANAlgorithm::Auto,
        leaf_size: 10,
        backend: SKBackendKind::Faer,
        execution_mode: SKExecutionMode::Automatic,
    };
    let training = SKTrainingData::from_view(SKDataView::Dense(data.view()));
    sk_run_dbscan_partition(&configuration, &training).unwrap()
}

#[test]
fn dbscan_fit_identifies_core_border_and_noise_points_on_hand_checked_fixture() {
    // Rows 0..4 form a 0.5x0.5 square (each has >= 4 neighbours within 1.0 -> core).
    // Row 4 lies within 1.0 of core row 0 and row 5 ({0, 4, 5}, 3 < 4 -> border of cluster 0).
    // Row 5 lies within 1.0 of border row 4 only ({4, 5}, 2 < 4 -> noise, no propagation).
    // Row 6 is isolated -> noise.
    let data = array![
        [0.0_f64, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [-0.9, 0.0],
        [-1.8, 0.0],
        [10.0, 10.0],
    ];
    let expansion = expand_dense(&data, 1.0, 4);

    assert_eq!(expansion.number_of_clusters, 1);
    assert_eq!(expansion.core_sample_indices.to_vec(), vec![0, 1, 2, 3]);
    assert_eq!(
        expansion.labels.to_vec(),
        vec![
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(0),
            SKClusterLabel::Noise,
            SKClusterLabel::Noise,
        ]
    );
}

#[test]
fn dbscan_fit_all_sparse_isolated_points_become_noise_with_zero_clusters() {
    let data = array![[0.0_f64, 0.0], [5.0, 5.0], [10.0, 10.0]];
    let expansion = expand_dense(&data, 0.5, 2);

    assert_eq!(expansion.number_of_clusters, 0);
    assert!(expansion.core_sample_indices.is_empty());
    assert_eq!(
        expansion.labels.to_vec(),
        vec![
            SKClusterLabel::Noise,
            SKClusterLabel::Noise,
            SKClusterLabel::Noise,
        ]
    );
}

#[test]
fn dbscan_fit_assigns_cluster_ids_in_smallest_core_index_order() {
    // Interleaved rows from two separated groups plus a leading noise row at index 0:
    // Group A (around 20.0) has its first core point at row 1 -> Cluster(0).
    // Group B (around 0.0) has its first core point at row 2 -> Cluster(1).
    let data = array![
        [100.0_f64, 100.0], // 0: noise
        [20.0, 20.0],       // 1: Group A core -> seeds Cluster(0)
        [0.0, 0.0],         // 2: Group B core -> seeds Cluster(1)
        [20.2, 20.0],       // 3: Group A core
        [0.2, 0.0],         // 4: Group B core
        [20.0, 20.2],       // 5: Group A core
        [0.0, 0.2],         // 6: Group B core
    ];
    let expansion = expand_dense(&data, 0.5, 3);

    assert_eq!(expansion.number_of_clusters, 2);
    assert_eq!(
        expansion.core_sample_indices.to_vec(),
        vec![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        expansion.labels.to_vec(),
        vec![
            SKClusterLabel::Noise,
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(1),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(1),
            SKClusterLabel::cluster(0),
            SKClusterLabel::cluster(1),
        ]
    );
}

#[test]
fn dbscan_fit_resolves_shared_border_point_to_earliest_seeded_cluster() {
    // Row 0 is a border point equidistant (0.9) from core row 1 (Cluster 0) and core row 5 (Cluster 1).
    // With minimum_samples = 4, row 0 has only 3 neighbours {0, 1, 5} so it is non-core,
    // while rows 1..5 form a 4-point core group and rows 5..9 form a 4-point core group.
    let data = array![
        [1.0_f64, 0.0], // 0: shared border point (neighbours: 0, 1, 5 -> 3 < 4)
        [0.1, 0.0],     // 1..5: first core group -> Cluster(0)
        [-0.3, 0.0],
        [-0.3, 0.3],
        [-0.3, -0.3],
        [1.9, 0.0], // 5..9: second core group -> Cluster(1)
        [2.3, 0.0],
        [2.3, 0.3],
        [2.3, -0.3],
    ];
    let expansion = expand_dense(&data, 1.0, 4);

    assert_eq!(expansion.number_of_clusters, 2);
    assert_eq!(expansion.labels[0], SKClusterLabel::cluster(0));
    assert_eq!(expansion.labels[1], SKClusterLabel::cluster(0));
    assert_eq!(expansion.labels[5], SKClusterLabel::cluster(1));
}

#[test]
fn dbscan_fit_repeated_runs_and_rayon_thread_counts_produce_identical_labels() {
    let data = Array2::from_shape_fn((180, 2), |(row, column)| {
        if row < 170 {
            let group = (row % 3) as f64 * 12.0;
            let offset = ((row * 7 + column * 3) % 5) as f64 * 0.08;
            group + offset
        } else {
            100.0 + (row as f64) * 5.0 + column as f64
        }
    });

    let baseline = expand_dense(&data, 0.5, 4);
    let repeated = expand_dense(&data, 0.5, 4);
    assert_eq!(baseline.strategy, super::SKDBSCANSearchStrategy::KdTree);
    assert_eq!(baseline.labels, repeated.labels);
    assert_eq!(baseline.core_sample_indices, repeated.core_sample_indices);
    assert_eq!(baseline.number_of_clusters, repeated.number_of_clusters);

    for threads in [1_usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let threaded = pool.install(|| expand_dense(&data, 0.5, 4));
        assert_eq!(threaded.labels, baseline.labels);
        assert_eq!(threaded.core_sample_indices, baseline.core_sample_indices);
        assert_eq!(threaded.number_of_clusters, baseline.number_of_clusters);
    }
}
