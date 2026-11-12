//! Tests for the Calinski-Harabasz index (spec `clustering-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKClusterLabel, SKDataView};

use super::sk_calinski_harabasz_score;

fn compact_separated_clusters() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ]
}

fn two_cluster_assignments() -> Array1<SKClusterLabel> {
    Array1::from_vec(vec![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ])
}

#[test]
fn calinski_harabasz_is_large_for_separated_clusters() {
    let features = compact_separated_clusters();
    let assignments = two_cluster_assignments();
    let score =
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert!(score > 100.0, "expected a large index, got {score}");
}

#[test]
fn calinski_harabasz_matches_the_hand_computed_fixture() {
    // Cluster 0 = {0, 1}, cluster 1 = {10, 11}: between dispersion 100,
    // within dispersion 1, so (100 / 1) / (1 / 2) = 200.
    let features: Array2<f64> = array![[0.0], [1.0], [10.0], [11.0]];
    let assignments = Array1::from_vec(vec![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ]);
    let score =
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert!((score - 200.0).abs() < 1e-12, "expected 200, got {score}");
}

#[test]
fn calinski_harabasz_excludes_noise_rows() {
    let mut features = compact_separated_clusters();
    features
        .push_row(array![1_000.0, 1_000.0].view())
        .expect("row fits");
    let mut assignments = two_cluster_assignments().to_vec();
    assignments.push(SKClusterLabel::NOISE);
    let with_noise = Array1::from_vec(assignments);

    let clean = sk_calinski_harabasz_score(
        SKDataView::Dense(features.slice(ndarray::s![0..6, ..])),
        two_cluster_assignments().view(),
    )
    .unwrap();
    let noisy =
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), with_noise.view()).unwrap();
    assert!((clean - noisy).abs() < 1e-12);
}

/// Coincident centroids leave no between-cluster dispersion: the ratio is
/// infinite as documented.
#[test]
fn calinski_harabasz_reports_infinity_for_coincident_centroids() {
    let features = ndarray::array![[0.0_f64], [0.0], [0.0], [0.0]];
    let assignments = ndarray::array![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ];
    let score =
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert!(score.is_infinite() && score > 0.0);
}
