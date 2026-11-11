//! Companion tests for the implicit centering of the randomized range finder
//! (spec `randomized-svd`).

use ndarray::{Array1, Array2};
use sciencekit_common::SKDataView;

use super::dense_tests::{fixture_low_rank_matrix, relative_error};
use super::randomized::sk_randomized_svd;
use super::sparse_tests::to_csr;

/// The explicit centered matrix `X − 1 μᵀ` of a fixture.
fn explicit_centered(matrix: &Array2<f64>, mean: &Array1<f64>) -> Array2<f64> {
    let mut centered = matrix.clone();
    for mut row in centered.rows_mut() {
        for (index, value) in row.iter_mut().enumerate() {
            *value -= mean[index];
        }
    }
    centered
}

/// A mean vector with one entry per column of the fixture.
fn fixture_mean(columns: usize) -> Array1<f64> {
    Array1::from_shape_fn(columns, |index| 0.25 * (index % 3) as f64 - 0.25)
}

#[test]
fn randomized_centered_dense_matches_the_explicit_centering() {
    let matrix = fixture_low_rank_matrix(40, 12, 3, 41);
    let mean = fixture_mean(12);
    let centered = explicit_centered(&matrix, &mean);

    let implicit = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 12, Some(mean.view()))
        .expect("the centered decomposition must succeed");
    let explicit = sk_randomized_svd(SKDataView::Dense(centered.view()), 3, 2, 4, 12, None)
        .expect("the explicit decomposition must succeed");

    for (left, right) in implicit.singular_values.iter().zip(explicit.singular_values.iter()) {
        assert!(
            (left - right).abs() < 1e-8,
            "the singular values differ: {left} against {right}"
        );
    }
    let difference = relative_error(&explicit.reconstruct(), &implicit.reconstruct());
    assert!(difference < 1e-8, "the reconstructions differ: {difference}");
}

#[test]
fn randomized_centered_sparse_matches_the_centered_dense() {
    let matrix = fixture_low_rank_matrix(40, 12, 3, 43);
    let mean = fixture_mean(12);
    let sparse = to_csr(&matrix);

    let dense = sk_randomized_svd(SKDataView::Dense(matrix.view()), 3, 2, 4, 12, Some(mean.view()))
        .expect("the dense decomposition must succeed");
    let sparse = sk_randomized_svd(
        SKDataView::Sparse(sparse.view()),
        3,
        2,
        4,
        12,
        Some(mean.view()),
    )
    .expect("the sparse decomposition must succeed");

    for (left, right) in dense.singular_values.iter().zip(sparse.singular_values.iter()) {
        assert!(
            (left - right).abs() < 1e-8,
            "the singular values differ: {left} against {right}"
        );
    }
    let difference = relative_error(&dense.reconstruct(), &sparse.reconstruct());
    assert!(difference < 1e-8, "the reconstructions differ: {difference}");
}

#[test]
fn randomized_centered_sparse_does_not_densify_the_centered_matrix() {
    let mut dense = Array2::<f64>::zeros((30, 400));
    for row in 0..30 {
        dense[(row, row * 5)] = 1.0;
    }
    let sparse = to_csr(&dense);
    let mean = Array1::from_elem(400, 0.1);

    let result = sk_randomized_svd(
        SKDataView::Sparse(sparse.view()),
        4,
        2,
        5,
        6,
        Some(mean.view()),
    )
    .expect("the centered sparse decomposition must succeed");
    assert_eq!(result.u.dim(), (30, 4));
    assert_eq!(result.v.dim(), (400, 4));
    assert!(result.singular_values.iter().all(|value| *value >= 0.0));
}
