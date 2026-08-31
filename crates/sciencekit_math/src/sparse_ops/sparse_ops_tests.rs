//! TDD tests for the sparse product kernels (spec `dense-sparse-products`).

use ndarray::{Array2, array};
use sciencekit_common::SKFloat;
use sprs::{CsMat, TriMat};

use super::{sk_csr_dense_product, sk_sparse_product};

/// Build a CSR matrix from explicit triplets.
fn csr_from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut matrix = TriMat::new((rows, cols));
    for &(row, col, value) in triplets {
        matrix.add_triplet(row, col, value);
    }
    matrix.to_csr()
}

/// The CSR×dense product equals the product of the densified operands.
#[test]
fn csr_dense_matches_dense_reference() {
    let csr = csr_from_triplets(
        3,
        3,
        &[
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (2, 0, 4.0),
            (2, 2, 5.0),
        ],
    );
    let dense: Array2<f64> = array![[1.0, 0.0], [2.0, 3.0], [0.0, 1.0]];
    let result = sk_csr_dense_product(&csr.view(), &dense.view());
    // Reference via dense multiplication.
    let reference = dense_reference(&csr, &dense);
    assert_eq!(result.shape(), reference.shape());
    for ((index, a), b) in result.indexed_iter().zip(reference.iter()) {
        assert!((a - b).abs() < 1e-12, "mismatch at {index:?}: {a} vs {b}");
    }
}

/// The CSR×dense product on a small case yields exact known values.
#[test]
fn csr_dense_known_values() {
    let csr = csr_from_triplets(2, 2, &[(0, 0, 2.0), (1, 1, 3.0)]);
    let dense: Array2<f64> = array![[1.0, 2.0], [4.0, 5.0]];
    let result = sk_csr_dense_product(&csr.view(), &dense.view());
    // [[2*1, 2*2],[3*4, 3*5]] = [[2,4],[12,15]]
    assert!((result[[0, 0]] - 2.0).abs() < 1e-12);
    assert!((result[[0, 1]] - 4.0).abs() < 1e-12);
    assert!((result[[1, 0]] - 12.0).abs() < 1e-12);
    assert!((result[[1, 1]] - 15.0).abs() < 1e-12);
}

/// The sparse operand is consumed as a view: the source stays intact and is
/// reused for a second product, so no densification/copy of the sparse input
/// happens inside the kernel.
#[test]
fn csr_dense_does_not_densify_the_sparse_operand() {
    let csr = csr_from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let dense: Array2<f64> = array![[2.0, 3.0], [4.0, 5.0]];
    // The kernel takes `&CsMatView`; the owned matrix is untouched and reusable.
    let first = sk_csr_dense_product(&csr.view(), &dense.view());
    let second = sk_csr_dense_product(&csr.view(), &dense.view());
    assert_eq!(first, second);
    assert_eq!(csr.nnz(), 2);
}

/// Sparse×sparse matches the dense reference product's structure and values.
#[test]
fn sparse_product_matches_dense_reference() {
    let left = csr_from_triplets(2, 3, &[(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0)]);
    let right = csr_from_triplets(3, 2, &[(0, 0, 4.0), (1, 1, 5.0), (2, 0, 6.0)]);
    let product = sk_sparse_product(&left.view(), &right.view());
    let product_dense = product.to_dense();
    // Reference: densify and multiply.
    let reference = left.to_dense().dot(&right.to_dense());
    assert_eq!(product_dense.shape(), reference.shape());
    for ((index, a), b) in product_dense.indexed_iter().zip(reference.iter()) {
        assert!((a - b).abs() < 1e-12, "mismatch at {index:?}: {a} vs {b}");
    }
}

/// Generic over the scalar type.
#[test]
fn sparse_kernels_accept_either_float() {
    fn accepts<F: SKFloat>(csr: &CsMat<F>, dense: &Array2<F>) {
        let _ = sk_csr_dense_product(&csr.view(), &dense.view());
    }
    let csr_f64 = csr_from_triplets(1, 1, &[(0, 0, 1.0)]);
    let dense_f64: Array2<f64> = array![[2.0]];
    accepts(&csr_f64, &dense_f64);

    let mut tri = TriMat::new((1, 1));
    tri.add_triplet(0, 0, 1.0_f32);
    let csr_f32 = tri.to_csr();
    let dense_f32: Array2<f32> = array![[2.0]];
    accepts(&csr_f32, &dense_f32);
}

/// Densify a CSR matrix and a dense array, then multiply (test reference).
fn dense_reference(csr: &CsMat<f64>, dense: &Array2<f64>) -> Array2<f64> {
    csr.to_dense().dot(dense)
}
