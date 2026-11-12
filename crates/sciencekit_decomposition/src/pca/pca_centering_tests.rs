//! Companion tests for PCA centering, feature means and explicit densification.

use ndarray::{Array2, Axis, array};
use sciencekit_common::{SKDataView, SKError};
use sprs::{CsMat, TriMat};

use super::centering::{
    sk_center_dense, sk_centered_dense_or_reject, sk_feature_means, sk_total_centered_variance,
};
use super::test_tracing::PcaCaptureGuard;

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
fn pca_centering_computes_hand_checked_means_and_zero_mean_centered_data() {
    let data = array![
        [1.0_f64, 10.0, -3.0],
        [3.0, 20.0, 0.0],
        [5.0, 30.0, 3.0],
        [7.0, 40.0, 8.0]
    ];
    let mean = sk_feature_means(SKDataView::Dense(data.view()), 1).unwrap();
    assert!((mean[0] - 4.0).abs() < 1e-12);
    assert!((mean[1] - 25.0).abs() < 1e-12);
    assert!((mean[2] - 2.0).abs() < 1e-12);

    let centered = sk_center_dense(data.view(), mean.view());
    let centered_column_means = centered.mean_axis(Axis(0)).unwrap();
    for &column_mean in centered_column_means.iter() {
        assert!(column_mean.abs() < 1e-12);
    }
}

#[test]
fn pca_centering_sparse_means_and_total_variance_match_dense_without_densifying() {
    let dense = array![
        [0.0_f64, 4.0, 0.0, -2.0],
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 6.0, 0.0],
        [0.0, 0.0, 0.0, 8.0]
    ];
    let sparse = csr_from_dense(&dense);

    let dense_mean = sk_feature_means(SKDataView::Dense(dense.view()), 1).unwrap();
    let sparse_mean = sk_feature_means(SKDataView::Sparse(sparse.view()), 1).unwrap();
    for (left, right) in dense_mean.iter().zip(sparse_mean.iter()) {
        assert!((left - right).abs() < 1e-12);
    }

    let dense_variance =
        sk_total_centered_variance(SKDataView::Dense(dense.view()), dense_mean.view()).unwrap();
    let sparse_variance =
        sk_total_centered_variance(SKDataView::Sparse(sparse.view()), sparse_mean.view()).unwrap();
    assert!((dense_variance - sparse_variance).abs() < 1e-12);
}

#[test]
fn pca_centering_rejects_sparse_by_default_and_warns_when_explicitly_densified() {
    let dense = array![[0.0_f64, 2.0, 0.0], [4.0, 0.0, 6.0]];
    let sparse = csr_from_dense(&dense);
    let mean = sk_feature_means(SKDataView::Sparse(sparse.view()), 1).unwrap();

    let rejected =
        sk_centered_dense_or_reject(SKDataView::Sparse(sparse.view()), mean.view(), false)
            .unwrap_err();
    match rejected {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csr");
            assert!(suggestion.contains("Randomized"));
            assert!(suggestion.contains("SKTruncatedSVD"));
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }

    let capture = PcaCaptureGuard::begin();
    let densified =
        sk_centered_dense_or_reject(SKDataView::Sparse(sparse.view()), mean.view(), true).unwrap();
    let warnings = capture.warning_events();
    assert_eq!(warnings.len(), 1);
    assert_eq!(
        warnings[0].get("estimated_dense_bytes").map(String::as_str),
        Some("48")
    );

    let expected = sk_center_dense(dense.view(), mean.view());
    for (left, right) in densified.iter().zip(expected.iter()) {
        assert!((left - right).abs() < 1e-12);
    }
}
