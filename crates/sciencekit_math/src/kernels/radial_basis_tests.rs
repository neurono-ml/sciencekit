//! Companion TDD tests for the radial-basis-function kernel
//! (spec `kernel-functions`).

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};
use sprs::TriMat;

use super::sk_radial_basis_function_kernel_matrix;

#[test]
fn radial_basis_function_kernel_handles_identical_and_distant_rows_and_cancellation() {
    let features = array![
        [1.0_f64, 2.0],
        [1.0, 2.0],
        [1000.0, -1000.0],
        [1.0e8 + 0.1234567, 1.0e8 + 0.7654321]
    ];

    let kernel = sk_radial_basis_function_kernel_matrix(
        SKDataView::Dense(features.view()),
        SKDataView::Dense(features.view()),
        0.5,
    )
    .unwrap();

    assert_eq!(kernel.dim(), (4, 4));
    assert!((kernel[[0, 0]] - 1.0).abs() < 1e-12);
    assert!((kernel[[0, 1]] - 1.0).abs() < 1e-12);
    // Large coordinate self-similarity must clamp tiny negative cancellation to 0 so exp <= 1.
    assert!((kernel[[3, 3]] - 1.0).abs() < 1e-12);
    assert!(kernel[[3, 3]] <= 1.0);
    // Distant rows underflow cleanly to 0 without NaN.
    assert!(kernel[[0, 2]].is_finite());
    assert!(kernel[[0, 2]].abs() < 1e-12);
}

#[test]
fn radial_basis_function_kernel_matches_across_dense_sparse_and_mixed_views() {
    let left_dense = array![[0.0_f64, 1.5, 0.0], [-2.0, 0.0, 3.0]];
    let right_dense = array![[0.0_f64, 1.5, 0.0], [1.0, -0.5, 0.0]];

    let to_csr = |dense: &Array2<f64>| {
        let mut triplets = TriMat::new(dense.dim());
        dense
            .indexed_iter()
            .filter(|(_, value)| **value != 0.0)
            .for_each(|((r, c), &v)| triplets.add_triplet(r, c, v));
        triplets.to_csr()
    };
    let left_csr = to_csr(&left_dense);
    let right_csr = to_csr(&right_dense);

    let dd = sk_radial_basis_function_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Dense(right_dense.view()),
        0.4,
    )
    .unwrap();
    let ss = sk_radial_basis_function_kernel_matrix(
        SKDataView::Sparse(left_csr.view()),
        SKDataView::Sparse(right_csr.view()),
        0.4,
    )
    .unwrap();
    let sd = sk_radial_basis_function_kernel_matrix(
        SKDataView::Sparse(left_csr.view()),
        SKDataView::Dense(right_dense.view()),
        0.4,
    )
    .unwrap();
    let ds = sk_radial_basis_function_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Sparse(right_csr.view()),
        0.4,
    )
    .unwrap();

    for candidate in [&ss, &sd, &ds] {
        candidate
            .iter()
            .zip(dd.iter())
            .for_each(|(&actual, &expected)| assert!((actual - expected).abs() < 1e-12));
    }

    let neg_gamma = sk_radial_basis_function_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Dense(right_dense.view()),
        -0.1,
    )
    .unwrap_err();
    assert!(matches!(
        neg_gamma,
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));
}
