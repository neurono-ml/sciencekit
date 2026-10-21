//! Retained-design accumulator tests (spec `coordinate-descent-solver`,
//! Decision 7).

use ndarray::s;
use sciencekit_common::SKError;

use super::accumulator::SKCoordinateDescentState;
use super::config::{SKCoordinateDescentConfig, SKSelection};
use super::test_datasets::{assert_close, orthogonal_design, to_csr};

/// A tight, screening-free configuration.
fn config() -> SKCoordinateDescentConfig<f64> {
    SKCoordinateDescentConfig {
        tolerance: 1e-10,
        maximum_number_of_iterations: 100_000,
        positive: false,
        selection: SKSelection::Cyclic,
        screening: false,
    }
}

/// Two dense batches accumulate to the concatenated fit.
#[test]
fn dense_accumulation_equals_concatenation() {
    let (features, targets) = orthogonal_design();
    let mut state = SKCoordinateDescentState::new_dense();
    state.accumulate_dense(features.slice(s![0..2, ..]), targets.slice(s![0..2]));
    state.accumulate_dense(features.slice(s![2.., ..]), targets.slice(s![2..]));
    assert_eq!(state.row_count(), 4);
    assert_eq!(state.columns_count(), 3);
    let solution = state.solve(&[0.5], 1.0, true, &config()).unwrap();
    assert_close(
        &solution.coefficients.row(0).to_vec(),
        &[1.5, -0.5, 0.0],
        1e-6,
    );
}

/// Two sparse batches accumulate to the concatenated fit.
#[test]
fn sparse_accumulation_equals_concatenation() {
    let (features, targets) = orthogonal_design();
    let first = to_csr(&features.slice(s![0..2, ..]).to_owned());
    let second = to_csr(&features.slice(s![2.., ..]).to_owned());
    let mut state = SKCoordinateDescentState::new_sparse();
    state.accumulate_sparse(first.view(), targets.slice(s![0..2]));
    state.accumulate_sparse(second.view(), targets.slice(s![2..]));
    assert_eq!(state.row_count(), 4);
    assert_eq!(state.columns_count(), 3);
    let solution = state.solve(&[0.5], 1.0, true, &config()).unwrap();
    assert_close(
        &solution.coefficients.row(0).to_vec(),
        &[1.5, -0.5, 0.0],
        1e-6,
    );
}

/// An empty accumulator refuses to solve.
#[test]
fn empty_accumulator_solve_is_rejected() {
    let state = SKCoordinateDescentState::new_dense();
    assert!(state.is_empty());
    let error = state.solve(&[1.0], 1.0, true, &config()).unwrap_err();
    assert!(matches!(error, SKError::Conversion(_)));
}
