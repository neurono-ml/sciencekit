//! TDD tests for the cross-product reduction kernels (spec
//! `higher-order-kernels`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::SKFloat;

use super::{sk_gram_matrix, sk_gram_vector};

/// Assert two `f64` vectors agree within a relative floating-point tolerance
/// (parallel reordering may shift the last bits of reductions).
fn assert_close_1d(a: &Array1<f64>, b: &Array1<f64>) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        let tolerance = 1e-9 * (1.0 + x.abs().max(y.abs()));
        assert!(
            (x - y).abs() <= tolerance,
            "values {x} and {y} differ beyond tolerance {tolerance}"
        );
    }
}

/// Assert two `f64` matrices agree element-wise within a relative tolerance.
fn assert_close_2d(a: &Array2<f64>, b: &Array2<f64>) {
    assert_eq!(a.dim(), b.dim());
    for ((i, j), x) in a.indexed_iter() {
        let y = b[(i, j)];
        let tolerance = 1e-9 * (1.0 + x.abs().max(y.abs()));
        assert!(
            (x - y).abs() <= tolerance,
            "values {x} and {y} at ({i}, {j}) differ beyond tolerance {tolerance}"
        );
    }
}

/// The Gram matrix is the matrix of column dot products on a hand-checkable
/// block.
#[test]
fn gram_matrix_matches_column_dot_products() {
    let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let gram = sk_gram_matrix(&input.view(), 1);
    assert_eq!(
        gram,
        array![[17.0, 22.0, 27.0], [22.0, 29.0, 36.0], [27.0, 36.0, 45.0]]
    );
}

/// The Gram vector is the vector of column-with-target dot products on a
/// hand-checkable block.
#[test]
fn gram_vector_matches_column_dot_targets() {
    let input: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let target = array![1.0, 1.0, 1.0];
    let gram = sk_gram_vector(&input.view(), &target.view(), 1);
    assert_eq!(gram, array![9.0, 12.0]);
}

/// Both kernels stay generic over the two supported floats.
#[test]
fn gram_kernels_accept_either_float() {
    fn accepts<F: SKFloat>(input: &Array2<F>, target: &Array1<F>) {
        let _ = sk_gram_matrix(&input.view(), 1);
        let _ = sk_gram_vector(&input.view(), &target.view(), 1);
    }
    accepts(&array![[1.0_f32, 2.0]], &array![1.0_f32]);
    accepts(&array![[1.0_f64, 2.0]], &array![1.0_f64]);
}

/// The Gram matrix is symmetric by construction.
#[test]
fn gram_matrix_is_symmetric() {
    let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
    let gram = sk_gram_matrix(&input.view(), 1);
    assert_eq!(gram, gram.t().to_owned());
}

/// The parallel Gram matrix matches the sequential reference on a large block.
#[test]
fn parallel_gram_matrix_matches_sequential() {
    let (rows, cols) = (5000, 12);
    let input =
        Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) as f64).sin() + 0.5);
    let sequential = sk_gram_matrix(&input.view(), 1);
    let parallel = sk_gram_matrix(&input.view(), 8);
    assert_close_2d(&sequential, &parallel);
}

/// The parallel Gram vector matches the sequential reference on a large block.
#[test]
fn parallel_gram_vector_matches_sequential() {
    let (rows, cols) = (5000, 12);
    let input =
        Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) as f64).cos() + 0.25);
    let target = Array1::<f64>::from_shape_fn(rows, |i| (i as f64) / 3.0 - 1.0);
    let sequential = sk_gram_vector(&input.view(), &target.view(), 1);
    let parallel = sk_gram_vector(&input.view(), &target.view(), 8);
    assert_close_1d(&sequential, &parallel);
}
