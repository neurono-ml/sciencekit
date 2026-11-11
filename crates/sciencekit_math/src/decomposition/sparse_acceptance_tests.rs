//! Acceptance tests of the sparse path of the randomized range finder
//! (PRD §8.7).

use ndarray::{Array1, Array2};
use sciencekit_common::SKDataView;

use super::randomized::sk_randomized_svd;
use super::sparse_tests::to_csr;

#[test]
fn randomized_acceptance_large_sparse_matrix_is_factorized() {
    let mut dense = Array2::<f64>::zeros((500, 1_000));
    for row in 0..500 {
        for offset in 0..4usize {
            dense[(row, (row * 3 + offset) % 1_000)] = 1.0 + offset as f64;
        }
    }
    let sparse = to_csr(&dense);
    let result = sk_randomized_svd(SKDataView::Sparse(sparse.view()), 6, 3, 8, 23, None)
        .expect("the decomposition must succeed");
    assert_eq!(result.u.dim(), (500, 6));
    assert_eq!(result.v.dim(), (1_000, 6));
    assert!(result.singular_values.iter().all(|value| *value >= 0.0));
}

#[test]
fn randomized_acceptance_centered_large_sparse_matrix_is_factorized() {
    let mut dense = Array2::<f64>::zeros((400, 800));
    for row in 0..400 {
        dense[(row, row % 800)] = 2.0;
    }
    let sparse = to_csr(&dense);
    let mean = Array1::<f64>::from_elem(800, 0.05);
    let result = sk_randomized_svd(
        SKDataView::Sparse(sparse.view()),
        5,
        2,
        6,
        31,
        Some(mean.view()),
    )
    .expect("the centered decomposition must succeed");
    assert_eq!(result.u.dim(), (400, 5));
    assert_eq!(result.v.dim(), (800, 5));
}
