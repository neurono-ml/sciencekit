//! Companion TDD tests for the precomputed kernel validation wrapper
//! (spec `kernel-functions`).

use ndarray::array;
use sciencekit_common::{SKDataView, SKError};
use sprs::TriMat;

use super::{sk_precomputed_kernel_matrix, sk_validate_precomputed_query_kernel};

#[test]
fn precomputed_kernel_accepts_valid_square_matrix_with_matching_rows() {
    let gram = array![[2.0_f64, 1.0, 0.0], [1.0, 3.0, -0.5], [0.0, -0.5, 4.0]];
    let validated = sk_precomputed_kernel_matrix(SKDataView::Dense(gram.view()), 3).unwrap();
    assert_eq!(validated, gram);

    let query = array![[1.0_f64, 0.5, -0.5], [0.0, 2.0, 1.0]];
    let query_view =
        sk_validate_precomputed_query_kernel(SKDataView::Dense(query.view()), 3).unwrap();
    assert_eq!(query_view.dim(), (2, 3));
}

#[test]
fn precomputed_kernel_rejects_non_square_row_mismatch_and_sparse_views() {
    let non_square = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let err = sk_precomputed_kernel_matrix(SKDataView::Dense(non_square.view()), 2).unwrap_err();
    assert!(matches!(err, SKError::ShapeMismatch { .. }));

    let square = array![[1.0_f64, 0.5], [0.5, 2.0]];
    let row_err = sk_precomputed_kernel_matrix(SKDataView::Dense(square.view()), 3).unwrap_err();
    assert!(matches!(row_err, SKError::ShapeMismatch { .. }));

    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 0, 1.0_f64);
    let csr = triplets.to_csr();
    let sparse_err = sk_precomputed_kernel_matrix(SKDataView::Sparse(csr.view()), 2).unwrap_err();
    assert!(matches!(
        sparse_err,
        SKError::UnsupportedRepresentation { .. }
    ));
}
