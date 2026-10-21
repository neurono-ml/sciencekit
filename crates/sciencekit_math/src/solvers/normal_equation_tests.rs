//! TDD tests for the zero-copy CSR normal-equation operator (spec
//! `iterative-linear-solvers`).

use ndarray::array;
use sciencekit_common::SKFloat;
use sprs::{CsMat, TriMat};

use super::SKCsrNormalEquationOperator;
use super::SKLinearOperator;
use super::solvers_tests::assert_close_1d;

/// Build a CSR matrix from explicit triplets.
fn csr_from_triplets<F: SKFloat>(
    rows: usize,
    cols: usize,
    triplets: &[(usize, usize, F)],
) -> CsMat<F> {
    let mut matrix = TriMat::new((rows, cols));
    for &(row, column, value) in triplets {
        matrix.add_triplet(row, column, value);
    }
    matrix.to_csr()
}

/// The sparse normal-equation operator equals the explicitly formed `AᵀA`
/// product on a vector.
#[test]
fn normal_equation_operator_matches_explicit_gram_product() {
    let matrix = csr_from_triplets(
        3,
        2,
        &[(0, 0, 1.0_f64), (0, 1, 2.0), (1, 0, 3.0), (2, 1, 4.0)],
    );
    let operator = SKCsrNormalEquationOperator::new(matrix.view());
    let vector = array![1.0_f64, -2.0];
    let result = operator.apply(vector.view());
    let dense = matrix.to_dense();
    let reference = dense.t().dot(&dense).dot(&vector);
    assert_close_1d(&result, &reference, 1e-12);
}

/// The normal-equation operator reports the column count as its dimension.
#[test]
fn normal_equation_operator_reports_column_dimension() {
    let matrix = csr_from_triplets(4, 3, &[(0, 0, 1.0_f64)]);
    let operator = SKCsrNormalEquationOperator::new(matrix.view());
    assert_eq!(operator.dimension(), 3);
}

/// The operator borrows the sparse view in place: repeated applications are
/// identical and the source matrix is untouched (no densification/copy).
#[test]
fn normal_equation_operator_borrows_the_view_in_place() {
    let matrix = csr_from_triplets(2, 2, &[(0, 0, 1.0_f64), (1, 1, 1.0)]);
    let operator = SKCsrNormalEquationOperator::new(matrix.view());
    let vector = array![2.0_f64, 3.0];
    let first = operator.apply(vector.view());
    let second = operator.apply(vector.view());
    assert_eq!(first, second);
    assert_eq!(matrix.nnz(), 2);
}

/// The normal-equation operator is generic over the two supported floats.
#[test]
fn normal_equation_operator_accepts_either_float() {
    let matrix64 = csr_from_triplets(2, 2, &[(0, 0, 1.0_f64), (1, 1, 2.0)]);
    let operator64 = SKCsrNormalEquationOperator::new(matrix64.view());
    let _ = operator64.apply(array![1.0_f64, 1.0].view());
    let matrix32 = csr_from_triplets(2, 2, &[(0, 0, 1.0_f32), (1, 1, 2.0_f32)]);
    let operator32 = SKCsrNormalEquationOperator::new(matrix32.view());
    let _ = operator32.apply(array![1.0_f32, 1.0].view());
}
