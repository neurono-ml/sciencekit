//! Shared train/test split fixtures.

use ndarray::Array2;

/// A dense design whose values identify their original row.
pub(crate) fn dense_features(number_of_rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((number_of_rows, 2), |(row, column)| {
        (row * 2 + column) as f64
    })
}
