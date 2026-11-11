//! Companion tests for DBSCAN core detection, expansion and deterministic
//! cluster identifiers (spec `dbscan`, decisions 2 and 3).

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKClusterLabel, SKDataView, SKExecutionMode};
use sciencekit_neighbors::SKTrainingData;

use super::config::SKDBSCANConfiguration;
use super::expansion::sk_expand_clusters;
use super::region_query::{sk_build_index, sk_radius_neighborhoods};
use super::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};

fn expand_dense(
    data: &Array2<f64>,
    eps: f64,
    minimum_samples: usize,
) -> super::expansion::SKDBSCANExpansion {
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
    let index = sk_build_index(&configuration, &training, SKDBSCANSearchStrategy::KdTree).unwrap();
    let neighborhoods = sk_radius_neighborhoods(&index, &training, eps).unwrap();
    sk_expand_clusters(&neighborhoods, minimum_samples).unwrap()
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
