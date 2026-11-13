//! Companion TDD tests for the sigmoid kernel (spec `kernel-functions`).

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};
use sprs::TriMat;

use super::sk_sigmoid_kernel_matrix;

#[test]
fn sigmoid_kernel_computes_hyperbolic_tangent_and_matches_all_representations() {
    let left_dense = array![[1.0_f64, 0.0, -2.0], [0.0, 3.0, 0.5]];
    let right_dense = array![[2.0_f64, 0.0, 1.0], [0.0, -1.0, 4.0]];

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

    let gamma = 0.25_f64;
    let coef0 = -0.5_f64;
    let dd = sk_sigmoid_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Dense(right_dense.view()),
        gamma,
        coef0,
    )
    .unwrap();
    let ss = sk_sigmoid_kernel_matrix(
        SKDataView::Sparse(left_csr.view()),
        SKDataView::Sparse(right_csr.view()),
        gamma,
        coef0,
    )
    .unwrap();
    let sd = sk_sigmoid_kernel_matrix(
        SKDataView::Sparse(left_csr.view()),
        SKDataView::Dense(right_dense.view()),
        gamma,
        coef0,
    )
    .unwrap();
    let ds = sk_sigmoid_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Sparse(right_csr.view()),
        gamma,
        coef0,
    )
    .unwrap();

    // Row 0 dot Right 0 = 2 - 2 = 0 => tanh(-0.5)
    assert!((dd[[0, 0]] - (-0.5_f64).tanh()).abs() < 1e-12);
    // Row 1 dot Right 1 = -3 + 2 = -1 => tanh(-0.25 - 0.5) = tanh(-0.75)
    assert!((dd[[1, 1]] - (-0.75_f64).tanh()).abs() < 1e-12);

    for candidate in [&ss, &sd, &ds] {
        candidate
            .iter()
            .zip(dd.iter())
            .for_each(|(&actual, &expected)| assert!((actual - expected).abs() < 1e-12));
    }

    let err = sk_sigmoid_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Dense(right_dense.view()),
        -1.0,
        0.0,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));
}
