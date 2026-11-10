//! Tests for the silhouette metric (spec `clustering-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKClusterLabel, SKDataView};
use sprs::CsMat;

use super::sk_silhouette_samples;

/// Two compact, well-separated clusters of three rows each.
fn two_compact_clusters() -> Array2<f64> {
    array![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ]
}

/// Assignments matching [`two_compact_clusters`].
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
fn silhouette_samples_of_separated_clusters_are_near_one() {
    let features = two_compact_clusters();
    let assignments = two_cluster_assignments();
    let samples =
        sk_silhouette_samples(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert_eq!(samples.len(), 6);
    for value in samples.iter() {
        assert!(*value > 0.9, "expected a near-one silhouette, got {value}");
        assert!(*value <= 1.0);
    }
}

#[test]
fn silhouette_samples_exclude_noise_rows() {
    let mut features = two_compact_clusters();
    features
        .push_row(array![100.0, 100.0].view())
        .expect("row fits");
    let mut assignments = two_cluster_assignments().to_vec();
    assignments.push(SKClusterLabel::NOISE);
    let with_noise = Array1::from_vec(assignments);

    let samples =
        sk_silhouette_samples(SKDataView::Dense(features.view()), with_noise.view()).unwrap();
    let clean = sk_silhouette_samples(
        SKDataView::Dense(features.slice(ndarray::s![0..6, ..])),
        two_cluster_assignments().view(),
    )
    .unwrap();

    assert_eq!(samples.len(), 6, "the noise row contributes no sample");
    for (noisy, expected) in samples.iter().zip(clean.iter()) {
        assert!((*noisy - *expected).abs() < 1e-12);
    }
}

#[test]
fn silhouette_samples_agree_between_dense_and_sparse() {
    let features = two_compact_clusters();
    let assignments = two_cluster_assignments();
    let dense =
        sk_silhouette_samples(SKDataView::Dense(features.view()), assignments.view()).unwrap();

    let sparse = CsMat::csr_from_dense(features.view(), 0.0);
    let sparse_view: SKDataView<'_, f64> = sparse.view().try_into().unwrap();
    let from_sparse = sk_silhouette_samples(sparse_view, assignments.view()).unwrap();

    assert_eq!(dense.len(), from_sparse.len());
    for (dense_value, sparse_value) in dense.iter().zip(from_sparse.iter()) {
        assert!((*dense_value - *sparse_value).abs() < 1e-12);
    }
}

#[test]
fn silhouette_samples_are_ordered_by_non_noise_row() {
    let features = two_compact_clusters();
    let assignments = Array1::from_vec(vec![
        SKClusterLabel::NOISE,
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::NOISE,
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ]);
    let samples =
        sk_silhouette_samples(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert_eq!(samples.len(), 4);
    for value in samples.iter() {
        assert!(*value > 0.9);
    }
}
