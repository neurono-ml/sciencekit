//! Companion tests for the randomized-SVD PCA solver.

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKDataView};
use sciencekit_math::backend::sk_math_backend_for_kind;
use sprs::{CsMat, TriMat};

use super::centering::{sk_center_dense, sk_feature_means, sk_total_centered_variance};
use super::solvers::{sk_solve_full_svd, sk_solve_randomized_svd};

fn planted_low_rank_design(rows: usize, columns: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, columns), |(row, column)| {
        let row_scalar = row as f64;
        let column_scalar = (column + 1) as f64;
        let factor_one = (0.17 * row_scalar + 0.5).sin() * (4.0 * column_scalar);
        let factor_two = (0.31 * row_scalar - 0.2).cos() * (1.8 * (columns - column) as f64);
        let factor_three = ((row % 7) as f64 - 3.0) * (0.4 * ((column % 3) as f64 + 1.0));
        factor_one + factor_two + factor_three + 2.5 * column_scalar
    })
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
fn pca_randomized_agrees_with_full_svd_on_low_rank_dense_design() {
    let data = planted_low_rank_design(48, 9);
    let mean = sk_feature_means(SKDataView::Dense(data.view()), 1).unwrap();
    let centered = sk_center_dense(data.view(), mean.view());
    let total_variance =
        sk_total_centered_variance(SKDataView::Dense(data.view()), mean.view()).unwrap();
    let backend = sk_math_backend_for_kind::<f64>(SKBackendKind::Faer);

    let full = sk_solve_full_svd(backend.as_ref(), centered.view(), 3, total_variance).unwrap();
    let randomized = sk_solve_randomized_svd(
        SKDataView::Dense(data.view()),
        mean.view(),
        3,
        5,
        17,
        SKBackendKind::Faer,
        total_variance,
    )
    .unwrap();

    for (left, right) in full
        .singular_values
        .iter()
        .zip(randomized.singular_values.iter())
    {
        assert!((left - right).abs() < 1e-6);
    }
    for (left, right) in full
        .explained_variance
        .iter()
        .zip(randomized.explained_variance.iter())
    {
        assert!((left - right).abs() < 1e-6);
    }
    for (left, right) in full
        .explained_variance_ratio
        .iter()
        .zip(randomized.explained_variance_ratio.iter())
    {
        assert!((left - right).abs() < 1e-7);
    }
    for (left, right) in full.components.iter().zip(randomized.components.iter()) {
        assert!((left - right).abs() < 1e-6);
    }
}

#[test]
fn pca_randomized_is_reproducible_and_matches_across_dense_and_sparse_views() {
    let mut dense = Array2::<f64>::zeros((36, 10));
    for row in 0..36 {
        dense[(row, row % 10)] = (row as f64 + 1.0) * 1.5;
        dense[(row, (row * 3 + 2) % 10)] = -((row % 5) as f64 + 0.75);
    }
    let sparse = csr_from_dense(&dense);
    let mean = sk_feature_means(SKDataView::Sparse(sparse.view()), 1).unwrap();
    let total_variance =
        sk_total_centered_variance(SKDataView::Sparse(sparse.view()), mean.view()).unwrap();

    let first = sk_solve_randomized_svd(
        SKDataView::Sparse(sparse.view()),
        mean.view(),
        3,
        5,
        42,
        SKBackendKind::Faer,
        total_variance,
    )
    .unwrap();
    let second = sk_solve_randomized_svd(
        SKDataView::Sparse(sparse.view()),
        mean.view(),
        3,
        5,
        42,
        SKBackendKind::Faer,
        total_variance,
    )
    .unwrap();
    assert_eq!(first.components, second.components);
    assert_eq!(first.singular_values, second.singular_values);

    let from_dense = sk_solve_randomized_svd(
        SKDataView::Dense(dense.view()),
        mean.view(),
        3,
        5,
        42,
        SKBackendKind::Faer,
        total_variance,
    )
    .unwrap();
    for (left, right) in first.components.iter().zip(from_dense.components.iter()) {
        assert!((left - right).abs() < 1e-10);
    }

    // Constant / zero-variance input still produces orthonormal component rows.
    let constant = Array2::from_elem((6, 3), 4.0_f64);
    let constant_mean = sk_feature_means(SKDataView::Dense(constant.view()), 1).unwrap();
    let constant_svd = sk_solve_randomized_svd(
        SKDataView::Dense(constant.view()),
        constant_mean.view(),
        3,
        3,
        5,
        SKBackendKind::Faer,
        0.0,
    )
    .unwrap();
    let gram = constant_svd.components.dot(&constant_svd.components.t());
    for ((row, column), &entry) in gram.indexed_iter() {
        let expected = if row == column { 1.0 } else { 0.0 };
        assert!((entry - expected).abs() < 1e-12);
    }
}
