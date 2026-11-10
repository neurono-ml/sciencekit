//! Shared sparse fixtures for the pairwise kernel tests.

use ndarray::{Array2, array};
use sprs::{CsMat, TriMat};

/// Build a canonical CSR matrix from explicit `(row, column, value)` triplets.
pub(crate) fn csr_from_triplets(
    rows: usize,
    columns: usize,
    triplets: &[(usize, usize, f64)],
) -> CsMat<f64> {
    let mut matrix = TriMat::new((rows, columns));
    for &(row, column, value) in triplets {
        matrix.add_triplet(row, column, value);
    }
    matrix.to_csr()
}

/// Build a CSR matrix from a dense array, treating exact zeros as implicit.
pub(crate) fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut matrix = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            matrix.add_triplet(row, column, value);
        }
    }
    matrix.to_csr()
}

/// A dense fixture with exact zeros so its CSR form stays sparse.
pub(crate) fn dense_fixture() -> Array2<f64> {
    array![
        [0.0, 1.0, 0.0, 2.0],
        [3.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-2.0, 0.0, 4.0, 0.5],
    ]
}

/// Assert two matrices agree element-wise within a tolerance.
pub(crate) fn assert_matrix_close(left: &Array2<f64>, right: &Array2<f64>, tolerance: f64) {
    assert_eq!(left.shape(), right.shape());
    for ((index, left_value), right_value) in left.indexed_iter().zip(right.iter()) {
        assert!(
            (left_value - right_value).abs() < tolerance,
            "mismatch at {index:?}: {left_value} vs {right_value}"
        );
    }
}
