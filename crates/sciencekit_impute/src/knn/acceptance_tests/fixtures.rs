//! Acceptance fixtures for the nearest-neighbor imputer.

use ndarray::Array2;

use crate::simple_imputer::SKImputedOutput;

/// Unwrap a dense transform result.
pub(super) fn dense(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => panic!("expected a dense result, found {matrix:?}"),
    }
}

/// A smooth design whose every column is a linear function of the row index,
/// so a nearest donor predicts a masked cell closely.
pub(super) fn linear_design(rows: usize, columns: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, columns), |(row, column)| {
        row as f64 * 0.5 * (column + 1) as f64
    })
}

/// Assert two scalars agree within the documented tolerance.
pub(super) fn assert_close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1e-9,
        "expected {right}, found {left}"
    );
}
