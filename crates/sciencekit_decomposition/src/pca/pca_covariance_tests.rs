//! Companion tests for the symmetric covariance eigensolver PCA path.

use ndarray::{Array2, array};
use sciencekit_common::{SKBackendKind, SKDataView};
use sciencekit_math::backend::sk_math_backend_for_kind;

use super::centering::{sk_center_dense, sk_feature_means, sk_total_centered_variance};
use super::solvers::{sk_solve_covariance_eigh, sk_solve_full_svd};

#[test]
fn pca_covariance_sorts_eigenvalues_descending_and_matches_full_svd() {
    let data = Array2::from_shape_fn((60, 5), |(row, column)| {
        let row_scalar = row as f64;
        let column_scalar = column as f64;
        3.5 * (0.23 * row_scalar + 0.1 * column_scalar).sin()
            + 1.7 * (0.41 * row_scalar - 0.5 * column_scalar).cos()
            + 0.6 * ((row % 5) as f64) * (column_scalar + 1.0)
            + 0.15 * ((row + 2 * column) % 7) as f64
    });
    let mean = sk_feature_means(SKDataView::Dense(data.view()), 1).unwrap();
    let centered = sk_center_dense(data.view(), mean.view());
    let total_variance =
        sk_total_centered_variance(SKDataView::Dense(data.view()), mean.view()).unwrap();
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let full = sk_solve_full_svd(backend.as_ref(), centered.view(), 4, total_variance).unwrap();
    let covariance =
        sk_solve_covariance_eigh(backend.as_ref(), centered.view(), 4, total_variance, 1).unwrap();

    for window in covariance.explained_variance.as_slice().unwrap().windows(2) {
        assert!(window[0] >= window[1]);
    }
    for (left, right) in full
        .explained_variance
        .iter()
        .zip(covariance.explained_variance.iter())
    {
        assert!((left - right).abs() < 1e-9);
    }
    for (left, right) in full
        .explained_variance_ratio
        .iter()
        .zip(covariance.explained_variance_ratio.iter())
    {
        assert!((left - right).abs() < 1e-10);
    }
    for (left, right) in full
        .singular_values
        .iter()
        .zip(covariance.singular_values.iter())
    {
        assert!((left - right).abs() < 1e-8);
    }
    for (left, right) in full.components.iter().zip(covariance.components.iter()) {
        assert!((left - right).abs() < 1e-9);
    }
}

#[test]
fn pca_covariance_clamps_tiny_negative_eigenvalues_on_rank_deficient_data() {
    let rank_one = Array2::from_shape_fn((12, 4), |(row, column)| {
        let factor = row as f64 - 5.5;
        factor * (column as f64 + 1.0)
    });
    let mean = sk_feature_means(SKDataView::Dense(rank_one.view()), 1).unwrap();
    let centered = sk_center_dense(rank_one.view(), mean.view());
    let total_variance =
        sk_total_centered_variance(SKDataView::Dense(rank_one.view()), mean.view()).unwrap();
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let result =
        sk_solve_covariance_eigh(backend.as_ref(), centered.view(), 4, total_variance, 1).unwrap();
    assert!(
        result
            .explained_variance
            .iter()
            .all(|&variance| variance.is_finite() && variance >= 0.0)
    );
    assert!(
        result
            .singular_values
            .iter()
            .all(|&singular| singular.is_finite() && singular >= 0.0)
    );

    let single_row = array![[2.0_f64, -4.0, 6.0]];
    let single_mean = sk_feature_means(SKDataView::Dense(single_row.view()), 1).unwrap();
    let single_centered = sk_center_dense(single_row.view(), single_mean.view());
    let single_result =
        sk_solve_covariance_eigh(backend.as_ref(), single_centered.view(), 1, 0.0, 1).unwrap();
    assert_eq!(single_result.explained_variance[0], 0.0);
    assert_eq!(single_result.singular_values[0], 0.0);
}
