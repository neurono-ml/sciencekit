//! Companion TDD tests for the polynomial kernel (spec `kernel-functions`).

use ndarray::array;
use sciencekit_common::{SKDataView, SKError};
use sprs::TriMat;

use super::{sk_linear_kernel_matrix, sk_polynomial_kernel_matrix};

#[test]
fn polynomial_kernel_computes_known_values_and_matches_sparse() {
    let left = array![[1.0_f64, 2.0], [3.0, -1.0]];
    let right = array![[2.0_f64, 0.5], [-1.0, 1.0]];

    let mut triplets = TriMat::new((2, 2));
    triplets.add_triplet(0, 0, 2.0_f64);
    triplets.add_triplet(0, 1, 0.5);
    triplets.add_triplet(1, 0, -1.0);
    triplets.add_triplet(1, 1, 1.0);
    let right_csr = triplets.to_csr();

    // Dots: [[3.0, 1.0], [5.5, -4.0]]
    // With gamma = 0.5, coef0 = 1.0, degree = 2:
    // (0.5 * dot + 1)^2 => [[(2.5)^2, (1.5)^2], [(3.75)^2, (-1.0)^2]]
    let dense = sk_polynomial_kernel_matrix(
        SKDataView::Dense(left.view()),
        SKDataView::Dense(right.view()),
        0.5,
        1.0,
        2,
    )
    .unwrap();
    let mixed = sk_polynomial_kernel_matrix(
        SKDataView::Dense(left.view()),
        SKDataView::Sparse(right_csr.view()),
        0.5,
        1.0,
        2,
    )
    .unwrap();

    let expected = array![[6.25_f64, 2.25], [14.0625, 1.0]];
    dense
        .iter()
        .zip(expected.iter())
        .for_each(|(&actual, &reference)| assert!((actual - reference).abs() < 1e-12));
    mixed
        .iter()
        .zip(expected.iter())
        .for_each(|(&actual, &reference)| assert!((actual - reference).abs() < 1e-12));
}

#[test]
fn polynomial_kernel_degree_one_with_unit_gamma_and_zero_offset_equals_linear() {
    let left = array![[1.0_f64, -2.0, 3.5], [0.5, 4.0, -1.0]];
    let right = array![[2.0_f64, 1.0, 0.0], [-3.0, 0.5, 2.0]];

    let linear = sk_linear_kernel_matrix(
        SKDataView::Dense(left.view()),
        SKDataView::Dense(right.view()),
    )
    .unwrap();
    let polynomial = sk_polynomial_kernel_matrix(
        SKDataView::Dense(left.view()),
        SKDataView::Dense(right.view()),
        1.0,
        0.0,
        1,
    )
    .unwrap();

    polynomial
        .iter()
        .zip(linear.iter())
        .for_each(|(&poly, &lin)| assert!((poly - lin).abs() < 1e-12));
}

#[test]
fn polynomial_kernel_rejects_non_finite_inputs_and_invalid_hyperparameters() {
    let valid = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let with_nan = array![[1.0_f64, 2.0], [f64::NAN, 4.0]];

    let nan_err = sk_polynomial_kernel_matrix(
        SKDataView::Dense(with_nan.view()),
        SKDataView::Dense(valid.view()),
        1.0,
        0.0,
        2,
    )
    .unwrap_err();
    assert!(matches!(nan_err, SKError::NonFiniteValue { position: 2 }));

    let degree_err = sk_polynomial_kernel_matrix(
        SKDataView::Dense(valid.view()),
        SKDataView::Dense(valid.view()),
        1.0,
        0.0,
        0,
    )
    .unwrap_err();
    assert!(matches!(
        degree_err,
        SKError::InvalidHyperparameter { name: "degree", .. }
    ));

    let gamma_err = sk_polynomial_kernel_matrix(
        SKDataView::Dense(valid.view()),
        SKDataView::Dense(valid.view()),
        -0.5,
        0.0,
        2,
    )
    .unwrap_err();
    assert!(matches!(
        gamma_err,
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));
}
