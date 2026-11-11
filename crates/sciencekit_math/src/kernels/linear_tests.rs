//! Companion TDD tests for the linear kernel (spec `kernel-functions`).

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};

use super::sk_linear_kernel_matrix;

#[test]
fn linear_kernel_computes_known_dense_dot_products_and_shape() {
    let left = array![[1.0_f64, 2.0, 3.0], [4.0, -1.0, 0.5]];
    let right = array![[2.0_f64, 0.0, -1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]];

    let kernel =
        sk_linear_kernel_matrix(SKDataView::Dense(left.view()), SKDataView::Dense(right.view()))
            .unwrap();

    assert_eq!(kernel.dim(), (2, 3));
    // Row 0: [2 - 3, 1 + 6 + 6, 2 + 12] = [-1, 13, 14]
    // Row 1: [8 - 0.5, 4 - 3 + 1, -1 + 2] = [7.5, 2, 1]
    let expected = array![[-1.0_f64, 13.0, 14.0], [7.5, 2.0, 1.0]];
    kernel
        .iter()
        .zip(expected.iter())
        .for_each(|(&actual, &reference)| assert!((actual - reference).abs() < 1e-12));
}

#[test]
fn linear_kernel_self_matrix_is_symmetric_and_rejects_feature_mismatch() {
    let features = array![[1.5_f64, -2.0], [0.5, 3.0], [-1.0, 4.0]];
    let gram = sk_linear_kernel_matrix(
        SKDataView::Dense(features.view()),
        SKDataView::Dense(features.view()),
    )
    .unwrap();

    assert_eq!(gram.dim(), (3, 3));
    gram.indexed_iter()
        .for_each(|((i, j), &value)| assert!((value - gram[[j, i]]).abs() < 1e-12));

    let mismatched = array![[1.0_f64, 2.0, 3.0]];
    let err = sk_linear_kernel_matrix(
        SKDataView::Dense(features.view()),
        SKDataView::Dense(mismatched.view()),
    )
    .unwrap_err();
    assert!(matches!(err, SKError::ShapeMismatch { .. }));
}

#[test]
fn linear_kernel_handles_large_dense_inputs() {
    let left = Array2::from_shape_fn((256, 32), |(r, c)| ((r + 1) * (c + 1)) as f64 * 0.01);
    let right = Array2::from_shape_fn((192, 32), |(r, c)| ((r + 2) as f64) * 0.05 - c as f64 * 0.01);

    let kernel =
        sk_linear_kernel_matrix(SKDataView::Dense(left.view()), SKDataView::Dense(right.view()))
            .unwrap();

    assert_eq!(kernel.dim(), (256, 192));
    let reference_cell: f64 = left
        .row(17)
        .iter()
        .zip(right.row(43).iter())
        .map(|(&a, &b)| a * b)
        .sum();
    assert!((kernel[[17, 43]] - reference_cell).abs() < 1e-9);
}
