//! Tests for the Davies-Bouldin index (spec `clustering-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKClusterLabel, SKDataView};

use super::sk_davies_bouldin_score;

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
fn davies_bouldin_is_near_zero_for_separated_clusters() {
    let features = compact_separated_clusters();
    let assignments = two_cluster_assignments();
    let score =
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert!(score < 0.2, "expected a near-zero index, got {score}");
    assert!(score >= 0.0);
}

#[test]
fn davies_bouldin_matches_the_hand_computed_fixture() {
    // Cluster 0 = {0, 1}, cluster 1 = {10, 11}: scatter 0.5 each, centroid
    // distance 10, so the index is (0.5 + 0.5) / 10 = 0.1.
    let features: Array2<f64> = array![[0.0], [1.0], [10.0], [11.0]];
    let assignments = Array1::from_vec(vec![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ]);
    let score =
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert!((score - 0.1).abs() < 1e-12, "expected 0.1, got {score}");
}

#[test]
fn davies_bouldin_excludes_noise_rows() {
    let mut features = compact_separated_clusters();
    features
        .push_row(array![1_000.0, 1_000.0].view())
        .expect("row fits");
    let mut assignments = two_cluster_assignments().to_vec();
    assignments.push(SKClusterLabel::NOISE);
    let with_noise = Array1::from_vec(assignments);

    let clean = sk_davies_bouldin_score(
        SKDataView::Dense(features.slice(ndarray::s![0..6, ..])),
        two_cluster_assignments().view(),
    )
    .unwrap();
    let noisy =
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), with_noise.view()).unwrap();
    assert!((clean - noisy).abs() < 1e-12);
}
