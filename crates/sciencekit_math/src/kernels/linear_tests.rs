//! Companion TDD tests for the linear kernel (spec `kernel-functions`).

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};

use super::sk_linear_kernel_matrix;

#[test]
fn linear_kernel_computes_known_dense_dot_products_and_shape() {
    let left = array![[1.0_f64, 2.0, 3.0], [4.0, -1.0, 0.5]];
    let right = array![[2.0_f64, 0.0, -1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]];

    let kernel = sk_linear_kernel_matrix(
        SKDataView::Dense(left.view()),
        SKDataView::Dense(right.view()),
    )
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
    let right = Array2::from_shape_fn((192, 32), |(r, c)| {
        ((r + 2) as f64) * 0.05 - c as f64 * 0.01
    });

    let kernel = sk_linear_kernel_matrix(
        SKDataView::Dense(left.view()),
        SKDataView::Dense(right.view()),
    )
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

#[test]
fn linear_kernel_matches_across_all_four_dense_and_sparse_pairs() {
    let left_dense = array![
        [0.0_f64, 2.0, 0.0, -3.0],
        [1.5, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ];
    let right_dense = array![[2.0_f64, 1.0, 0.0, 0.5], [0.0, -2.0, 3.0, 4.0]];

    let to_csr = |dense: &Array2<f64>| {
        let mut triplets = sprs::TriMat::new(dense.dim());
        dense
            .indexed_iter()
            .filter(|(_, value)| **value != 0.0)
            .for_each(|((r, c), &v)| triplets.add_triplet(r, c, v));
        triplets.to_csr()
    };
    let left_csr = to_csr(&left_dense);
    let right_csr = to_csr(&right_dense);

    let dd = sk_linear_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Dense(right_dense.view()),
    )
    .unwrap();
    let ss = sk_linear_kernel_matrix(
        SKDataView::Sparse(left_csr.view()),
        SKDataView::Sparse(right_csr.view()),
    )
    .unwrap();
    let sd = sk_linear_kernel_matrix(
        SKDataView::Sparse(left_csr.view()),
        SKDataView::Dense(right_dense.view()),
    )
    .unwrap();
    let ds = sk_linear_kernel_matrix(
        SKDataView::Dense(left_dense.view()),
        SKDataView::Sparse(right_csr.view()),
    )
    .unwrap();

    for candidate in [&ss, &sd, &ds] {
        assert_eq!(candidate.dim(), dd.dim());
        candidate
            .iter()
            .zip(dd.iter())
            .for_each(|(&actual, &expected)| assert!((actual - expected).abs() < 1e-12));
    }
}

#[test]
fn linear_kernel_handles_ultra_wide_csr_without_densifying() {
    // 4 rows by 100,000,000 columns: densifying would allocate 3.2 GB per matrix.
    let left = sprs::CsMat::new(
        (4, 100_000_000),
        vec![0, 2, 4, 5, 7],
        vec![3, 99_999_991, 10, 50_000_000, 99_999_991, 3, 50_000_000],
        vec![2.0_f64, 3.0, -1.0, 4.0, 5.0, 7.0, -2.0],
    );
    let right = sprs::CsMat::new(
        (2, 100_000_000),
        vec![0, 2, 4],
        vec![3, 50_000_000, 10, 99_999_991],
        vec![1.5_f64, 2.5, -3.0, 4.0],
    );

    let kernel = sk_linear_kernel_matrix(
        SKDataView::Sparse(left.view()),
        SKDataView::Sparse(right.view()),
    )
    .unwrap();

    assert_eq!(kernel.dim(), (4, 2));
    // Row 0 = {3: 2, 99_999_991: 3}; Right 0 = {3: 1.5, 50_000_000: 2.5} => 3.0
    // Row 0 dot Right 1 ({10: -3, 99_999_991: 4}) => 12.0
    assert!((kernel[[0, 0]] - 3.0).abs() < 1e-12);
    assert!((kernel[[0, 1]] - 12.0).abs() < 1e-12);
    assert!((kernel[[1, 0]] - 10.0).abs() < 1e-12);
    assert!((kernel[[1, 1]] - 3.0).abs() < 1e-12);
    assert!((kernel[[2, 1]] - 20.0).abs() < 1e-12);
    assert!((kernel[[3, 0]] - 5.5).abs() < 1e-12);
}
