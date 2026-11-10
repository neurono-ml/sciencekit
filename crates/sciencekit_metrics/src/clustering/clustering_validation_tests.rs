//! Validation tests for the clustering metrics (spec `clustering-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKClusterLabel, SKDataView, SKError};

use super::{
    SKMutualInformationNormalization, sk_adjusted_rand_score, sk_calinski_harabasz_score,
    sk_davies_bouldin_score, sk_homogeneity_score, sk_normalized_mutual_information_score,
    sk_silhouette_score,
};

fn features() -> Array2<f64> {
    array![[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]]
}

fn assignments() -> Array1<SKClusterLabel> {
    Array1::from_vec(vec![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ])
}

#[test]
fn clustering_validation_rejects_empty_input() {
    let empty: Array2<f64> = Array2::zeros((0, 2));
    let no_labels: Array1<SKClusterLabel> = Array1::from_vec(Vec::new());
    for error in [
        sk_silhouette_score(SKDataView::Dense(empty.view()), no_labels.view()).unwrap_err(),
        sk_davies_bouldin_score(SKDataView::Dense(empty.view()), no_labels.view()).unwrap_err(),
        sk_calinski_harabasz_score(SKDataView::Dense(empty.view()), no_labels.view()).unwrap_err(),
    ] {
        assert!(matches!(error, SKError::EmptyInput { .. }));
    }

    let empty_labels = Array1::<i64>::from_vec(Vec::new());
    assert!(matches!(
        sk_adjusted_rand_score(empty_labels.view(), empty_labels.view()).unwrap_err(),
        SKError::EmptyInput { .. }
    ));
}

#[test]
fn clustering_validation_rejects_length_mismatch() {
    let features = features();
    let short: Array1<SKClusterLabel> = Array1::from_vec(vec![SKClusterLabel::cluster(0); 3]);
    for error in [
        sk_silhouette_score(SKDataView::Dense(features.view()), short.view()).unwrap_err(),
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), short.view()).unwrap_err(),
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), short.view()).unwrap_err(),
    ] {
        assert!(matches!(error, SKError::ShapeMismatch { .. }));
    }

    let truth = array![0_i64, 0, 1, 1];
    let short_labels = array![0_i64, 0, 1];
    assert!(matches!(
        sk_homogeneity_score(truth.view(), short_labels.view()).unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));
}

#[test]
fn clustering_validation_rejects_a_single_cluster() {
    let features = features();
    let single = Array1::from_vec(vec![SKClusterLabel::cluster(2); 4]);
    for error in [
        sk_silhouette_score(SKDataView::Dense(features.view()), single.view()).unwrap_err(),
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), single.view()).unwrap_err(),
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), single.view()).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "assignments",
                ..
            }
        ));
    }
}

#[test]
fn clustering_validation_rejects_noise_only_input() {
    let features = features();
    let noise = Array1::from_vec(vec![SKClusterLabel::NOISE; 4]);
    for error in [
        sk_silhouette_score(SKDataView::Dense(features.view()), noise.view()).unwrap_err(),
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), noise.view()).unwrap_err(),
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), noise.view()).unwrap_err(),
    ] {
        assert!(matches!(error, SKError::EmptyInput { .. }));
    }
}

#[test]
fn clustering_validation_rejects_non_finite_features() {
    let mut dense = features();
    dense[[1, 0]] = f64::NAN;
    let error =
        sk_davies_bouldin_score(SKDataView::Dense(dense.view()), assignments().view()).unwrap_err();
    assert!(matches!(error, SKError::NonFiniteValue { .. }));

    let mut sparse = sprs::TriMat::new((4, 2));
    sparse.add_triplet(0, 0, 1.0_f64);
    sparse.add_triplet(1, 0, f64::INFINITY);
    sparse.add_triplet(2, 0, 10.0_f64);
    sparse.add_triplet(3, 0, 11.0_f64);
    let sparse: sprs::CsMat<f64> = sparse.to_csr();
    let sparse_view: SKDataView<'_, f64> = sparse.view().try_into().unwrap();
    let error = sk_calinski_harabasz_score(sparse_view, assignments().view()).unwrap_err();
    assert!(matches!(error, SKError::NonFiniteValue { .. }));
}

#[test]
fn clustering_validation_rejects_no_rows_beyond_the_clusters() {
    // Two rows and two clusters leave no within-cluster degrees of freedom.
    let features: Array2<f64> = array![[0.0], [10.0]];
    let assignments =
        Array1::from_vec(vec![SKClusterLabel::cluster(0), SKClusterLabel::cluster(1)]);
    let error = sk_calinski_harabasz_score(SKDataView::Dense(features.view()), assignments.view())
        .unwrap_err();
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "assignments",
            ..
        }
    ));
}

#[test]
fn clustering_validation_rejects_empty_pairwise_labels() {
    let labels = array![0_i64, 1, 0, 1];
    assert!(matches!(
        sk_normalized_mutual_information_score(
            labels.view(),
            array![0_i64, 1, 1].view(),
            SKMutualInformationNormalization::Arithmetic,
        )
        .unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));
}
