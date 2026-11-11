//! Companion tests for the sparse path of the randomized range finder (spec
//! `randomized-svd`).

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sprs::CsMat;

use super::dense_tests::{fixture_low_rank_matrix, relative_error};
use super::instrumentation::{sk_largest_dense_intermediate, sk_reset_dense_intermediate};
use super::randomized::sk_randomized_svd;

/// Builds the CSR matrix of a dense fixture.
fn to_csr(matrix: &Array2<f64>) -> CsMat<f64> {
    let (rows, columns) = matrix.dim();
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            let value = matrix[(row, column)];
            if value != 0.0 {
                indices.push(column);
                data.push(value);
            }
        }
        indptr.push(indices.len());
    }
    CsMat::new((rows, columns), indptr, indices, data)
}

#[test]
fn randomized_sparse_matches_the_dense_factorization() {
    let matrix = fixture_low_rank_matrix(40, 15, 4, 31);
    let sparse = to_csr(&matrix);
    let dense = sk_randomized_svd(SKDataView::Dense(matrix.view()), 4, 2, 5, 6, None)
        .expect("the dense decomposition must succeed");
    let sparse = sk_randomized_svd(SKDataView::Sparse(sparse.view()), 4, 2, 5, 6, None)
        .expect("the sparse decomposition must succeed");

    assert_eq!(dense.u.dim(), sparse.u.dim());
    assert_eq!(dense.v.dim(), sparse.v.dim());
    for (left, right) in dense.singular_values.iter().zip(sparse.singular_values.iter()) {
        assert!(
            (left - right).abs() < 1e-8,
            "the singular values differ: {left} against {right}"
        );
    }
    let dense_reconstruction = dense.reconstruct();
    let sparse_reconstruction = sparse.reconstruct();
    let difference = relative_error(&dense_reconstruction, &sparse_reconstruction);
    assert!(difference < 1e-8, "the reconstructions differ: {difference}");
}

#[test]
fn randomized_sparse_does_not_densify_a_wide_matrix() {
    let mut dense = Array2::<f64>::zeros((20, 500));
    for row in 0..20 {
        dense[(row, row * 7)] = 1.0;
        dense[(row, row * 7 + 1)] = 2.0;
    }
    let sparse = to_csr(&dense);
    let result = sk_randomized_svd(SKDataView::Sparse(sparse.view()), 5, 2, 4, 9, None)
        .expect("the sparse decomposition must succeed");
    assert_eq!(result.u.dim(), (20, 5));
    assert_eq!(result.v.dim(), (500, 5));
    assert!(result.singular_values.iter().all(|value| *value >= 0.0));
}

#[test]
fn randomized_sparse_reconstructs_a_low_rank_matrix() {
    let matrix = fixture_low_rank_matrix(50, 18, 3, 37);
    let sparse = to_csr(&matrix);
    let result = sk_randomized_svd(SKDataView::Sparse(sparse.view()), 3, 2, 5, 4, None)
        .expect("the sparse decomposition must succeed");
    let error = relative_error(&matrix, &result.reconstruct());
    assert!(error < 1e-8, "reconstruction error {error} is too large");
}

#[test]
fn randomized_sparse_keeps_the_dense_intermediates_small_on_a_large_matrix() {
    let (rows, columns) = (4_000usize, 20_000usize);
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..rows {
        for offset in 0..5usize {
            indices.push((row * 5 + offset) % columns);
            data.push(1.0 + offset as f64);
        }
        indptr.push(indices.len());
    }
    let sparse = CsMat::new((rows, columns), indptr, indices, data);

    sk_reset_dense_intermediate();
    let components = 5usize;
    let oversampling = 10usize;
    let result = sk_randomized_svd(
        SKDataView::Sparse(sparse.view()),
        components,
        2,
        oversampling,
        13,
        None,
    )
    .expect("the sparse decomposition must succeed");

    let basis_size = components + oversampling;
    let largest = sk_largest_dense_intermediate();
    let allowed = basis_size * rows.max(columns);
    assert!(
        largest <= allowed,
        "the dense intermediate reached {largest} cells, above the allowed {allowed}"
    );
    assert!(
        largest < rows * columns,
        "a dense copy of the input was materialized: {largest} cells"
    );
    assert_eq!(result.u.dim(), (rows, components));
    assert_eq!(result.v.dim(), (columns, components));
}
