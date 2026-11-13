//! Tests for the silhouette metric (spec `clustering-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKClusterLabel, SKDataView};
use sprs::CsMat;

use super::{sk_silhouette_samples, sk_silhouette_score};
use sciencekit_common::SKError;

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

#[test]
fn silhouette_score_is_the_mean_of_the_samples() {
    let features = two_compact_clusters();
    let assignments = two_cluster_assignments();
    let samples =
        sk_silhouette_samples(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    let expected = samples.iter().sum::<f64>() / samples.len() as f64;
    let score =
        sk_silhouette_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert!((score - expected).abs() < 1e-12);
    assert!(score > 0.9, "expected a near-one score, got {score}");
}

#[test]
fn silhouette_score_rejects_fewer_than_two_clusters() {
    let features = two_compact_clusters();
    let assignments = Array1::from_vec(vec![SKClusterLabel::cluster(3); 6]);
    let error =
        sk_silhouette_score(SKDataView::Dense(features.view()), assignments.view()).unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "assignments",
            ..
        }
    ));
}

#[test]
fn silhouette_score_rejects_all_noise_assignments() {
    let features = two_compact_clusters();
    let assignments = Array1::from_vec(vec![SKClusterLabel::NOISE; 6]);
    let error =
        sk_silhouette_score(SKDataView::Dense(features.view()), assignments.view()).unwrap_err();
    assert!(matches!(error, SKError::EmptyInput { .. }));
}

/// A hand-computed fixture pins the exact formula, including that `a(i)`
/// excludes the point itself.
#[test]
fn silhouette_samples_match_a_hand_computed_fixture() {
    // Four points on a line: cluster 0 is {0, 2}, cluster 1 is {10, 12}.
    let features = ndarray::array![[0.0_f64], [2.0], [10.0], [12.0]];
    let assignments = ndarray::array![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1)
    ];
    let samples =
        sk_silhouette_samples(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    // Point 0: a = d(0,2) = 2, b = mean(d(0,10), d(0,12)) = 11, s = 9/11.
    assert!((samples[0] - 9.0 / 11.0).abs() < 1e-12);
    // Point 2: a = 2, b = mean(d(2,10), d(2,12)) = 9, s = 7/9.
    assert!((samples[1] - 7.0 / 9.0).abs() < 1e-12);
    // Point 10: a = 2, b = mean(d(10,0), d(10,2)) = 9, s = 7/9.
    assert!((samples[2] - 7.0 / 9.0).abs() < 1e-12);
    // Point 12: a = 2, b = mean(d(12,0), d(12,2)) = 11, s = 9/11.
    assert!((samples[3] - 9.0 / 11.0).abs() < 1e-12);
}

/// A singleton cluster scores exactly zero by the documented convention.
#[test]
fn silhouette_samples_score_a_singleton_cluster_as_zero() {
    let features = ndarray::array![[0.0_f64], [10.0], [12.0]];
    let assignments = ndarray::array![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ];
    let samples =
        sk_silhouette_samples(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    assert_eq!(samples[0], 0.0);
}
