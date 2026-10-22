//! Companion tests for the logistic design abstraction and objective primitives
//! (spec `logistic-solver`).

use ndarray::{Array1, Array2, array};
use sprs::TriMat;

use super::design::{SKDenseDesign, SKLogisticDesign, SKSparseDesign};
use super::objective::{
    sk_binary_gradient, sk_binary_loss, sk_log_sigmoid, sk_sigmoid, sk_softmax,
};

fn dense() -> Array2<f64> {
    array![[1.0, 0.0, 2.0], [0.0, 3.0, -1.0], [4.0, 1.0, 0.0]]
}

fn to_csr(dense: &Array2<f64>) -> sprs::CsMat<f64> {
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense.indexed_iter().for_each(|((row, column), &value)| {
        if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

#[test]
fn dense_and_sparse_dot_all_agree() {
    let dense = dense();
    let sparse = to_csr(&dense);
    let weights = array![0.5, -1.0, 2.0];
    let mut dense_out = Array1::<f64>::zeros(3);
    let mut sparse_out = Array1::<f64>::zeros(3);
    SKDenseDesign::new(dense.view()).dot_all(weights.view(), dense_out.view_mut());
    SKSparseDesign::new(sparse.view()).dot_all(weights.view(), sparse_out.view_mut());
    for (dense_value, sparse_value) in dense_out.iter().zip(sparse_out.iter()) {
        assert!((dense_value - sparse_value).abs() < 1e-12);
    }
}

#[test]
fn dense_and_sparse_row_dot_and_add_agree() {
    let dense = dense();
    let sparse = to_csr(&dense);
    let weights = array![1.5, 0.25, -2.0];
    let dense_design = SKDenseDesign::new(dense.view());
    let sparse_design = SKSparseDesign::new(sparse.view());
    for row in 0..3 {
        let dense_dot = dense_design.row_dot(row, weights.view());
        let sparse_dot = sparse_design.row_dot(row, weights.view());
        assert!((dense_dot - sparse_dot).abs() < 1e-12);

        let mut dense_accumulator = Array1::<f64>::zeros(3);
        let mut sparse_accumulator = Array1::<f64>::zeros(3);
        dense_design.add_row_scaled(row, 2.0, dense_accumulator.view_mut());
        sparse_design.add_row_scaled(row, 2.0, sparse_accumulator.view_mut());
        for (dense_value, sparse_value) in dense_accumulator.iter().zip(sparse_accumulator.iter()) {
            assert!((dense_value - sparse_value).abs() < 1e-12);
        }
    }
}

#[test]
fn a_zero_row_contributes_nothing() {
    let zero = array![[0.0, 0.0, 0.0]];
    let zero_design = SKDenseDesign::new(zero.view());
    let mut out = Array1::<f64>::zeros(3);
    zero_design.add_row_scaled(0, 5.0, out.view_mut());
    assert!(out.iter().all(|value| *value == 0.0));

    let sparse_zero = to_csr(&zero);
    let sparse_design = SKSparseDesign::new(sparse_zero.view());
    let mut sparse_out = Array1::<f64>::zeros(3);
    sparse_design.add_row_scaled(0, 5.0, sparse_out.view_mut());
    assert!(sparse_out.iter().all(|value| *value == 0.0));
}

#[test]
fn sigmoid_is_stable_and_centered() {
    assert!((sk_sigmoid(0.0_f64) - 0.5).abs() < 1e-12);
    assert!(sk_sigmoid(1000.0_f64) == 1.0);
    assert!(sk_sigmoid(-1000.0_f64) == 0.0);
    for value in [-50.0_f64, -1.0, -0.1, 0.1, 1.0, 50.0] {
        let sigmoid = sk_sigmoid(value);
        assert!((sigmoid + sk_sigmoid(-value) - 1.0).abs() < 1e-12);
    }
}

#[test]
fn log_sigmoid_is_exact_in_the_tails() {
    assert!(sk_log_sigmoid(1000.0_f64).abs() < 1e-9);
    assert!((sk_log_sigmoid(-1000.0_f64) + 1000.0).abs() < 1e-9);
    assert!(sk_log_sigmoid(-1000.0_f64).is_finite());
}

#[test]
fn softmax_is_a_distribution() {
    let logits = [1.0_f64, 2.0, 3.0, -5.0];
    let mut out = [0.0_f64; 4];
    sk_softmax(&logits, &mut out);
    let total: f64 = out.iter().sum();
    assert!((total - 1.0).abs() < 1e-12);
    assert!(out.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    // The largest logit receives the largest probability.
    assert!(out[2] > out[0] && out[2] > out[3]);
}

#[test]
fn binary_gradient_is_the_derivative_of_the_loss() {
    // Central finite differences of the loss match the analytic gradient.
    let label = 1.0_f64;
    let weight = 2.5_f64;
    let point = 0.7_f64;
    let h = 1e-6;
    let numeric = (sk_binary_loss(point + h, label, weight)
        - sk_binary_loss(point - h, label, weight))
        / (2.0 * h);
    let analytic = sk_binary_gradient(point, label, weight);
    assert!((numeric - analytic).abs() < 1e-6);
}
