//! Sparse ridge prediction, multi-alpha and float-coverage tests
//! (spec `ridge-regression`).

use ndarray::array;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKRegressorPredictor, SKSupervisedFit};
use sprs::TriMat;

use super::builder::SKRidgeSolver;
use super::core_implementation::SKRidge;
use super::sparse_test_datasets::{sparse_cg, to_csr};
use super::test_datasets::{assert_close, assert_close_scalar, one_feature, two_feature};

/// A sequence of `alpha` yields one sparse solution per value, in order.
#[test]
fn sparse_alpha_sequence_yields_one_solution_per_value() {
    let (dense, targets) = one_feature();
    let sparse = to_csr(&dense);
    let model = SKRidge::build()
        .alpha(vec![0.0, 1.0, 4.0])
        .solver(SKRidgeSolver::SparseCg)
        .tolerance(1e-12)
        .maximum_number_of_iterations(10_000)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    assert_eq!(model.alphas(), &[0.0, 1.0, 4.0]);
    assert_eq!(model.coefficients().nrows(), 3);
    assert!(model.coefficient_for(0)[0] > model.coefficient_for(1)[0]);
    assert!(model.coefficient_for(1)[0] > model.coefficient_for(2)[0]);
}

/// The sparse path fits both supported floats.
#[test]
fn sparse_fits_both_supported_floats() {
    let targets = array![2.0f32, 4.0, 6.0];
    let mut triplets = TriMat::new((3, 1));
    triplets.add_triplet(0, 0, 1.0_f32);
    triplets.add_triplet(1, 0, 2.0_f32);
    triplets.add_triplet(2, 0, 3.0_f32);
    let sparse = triplets.to_csr();
    let model = SKRidge::<f32>::build()
        .alpha(2.0f32)
        .include_intercept(false)
        .solver(SKRidgeSolver::SparseCg)
        .tolerance(1e-6)
        .maximum_number_of_iterations(10_000)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    assert!((model.coefficient_for(0)[0] - 1.75).abs() < 1e-3);
}

/// Sparse and dense prediction agree for the same model.
#[test]
fn sparse_prediction_matches_dense() {
    let (dense, targets) = two_feature();
    let sparse = to_csr(&dense);
    let model = sparse_cg(1.0).fit(sparse.view(), targets.view()).unwrap();
    let dense_prediction = model.predict_with_alpha_index(0, dense.view()).unwrap();
    let sparse_prediction = model.predict_with_alpha_index(0, sparse.view()).unwrap();
    assert_close(
        &sparse_prediction.to_vec(),
        &dense_prediction.to_vec(),
        1e-9,
    );
}

/// Without an intercept the sparse fit solves through the origin.
#[test]
fn sparse_include_intercept_false() {
    let (dense, targets) = one_feature();
    let sparse = to_csr(&dense);
    let model = SKRidge::build()
        .alpha(2.0)
        .include_intercept(false)
        .solver(SKRidgeSolver::SparseCg)
        .tolerance(1e-12)
        .maximum_number_of_iterations(10_000)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    // Σxy = 28, Σx² = 14, α = 2 → 28 / 16 = 1.75.
    assert_close(&model.coefficient_for(0).to_vec(), &[1.75], 1e-6);
    assert_close_scalar(model.intercept_for(0), 0.0, 1e-12);
}

/// Prediction on a matrix with the wrong width is rejected.
#[test]
fn sparse_prediction_rejects_shape_mismatch() {
    let (dense, targets) = one_feature();
    let sparse = to_csr(&dense);
    let model = sparse_cg(1.0).fit(sparse.view(), targets.view()).unwrap();
    let wrong = array![[1.0, 2.0]];
    let error = model.predict(wrong.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
