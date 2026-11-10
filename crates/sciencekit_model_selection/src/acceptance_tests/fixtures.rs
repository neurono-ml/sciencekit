//! Deterministic acceptance fixtures.

use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};

/// A deterministic dense design with distinct, finite values.
pub(crate) fn dense_design(number_of_rows: usize, number_of_columns: usize) -> Array2<f64> {
    Array2::from_shape_fn((number_of_rows, number_of_columns), |(row, column)| {
        ((row * 7 + column * 3) % 101) as f64 / 7.0
    })
}

/// Cyclic class labels covering `number_of_classes` classes.
pub(crate) fn cyclic_labels(number_of_rows: usize, number_of_classes: i64) -> Array1<i64> {
    Array1::from_shape_fn(number_of_rows, |row| (row as i64) % number_of_classes)
}

/// Contiguous groups, each holding an equal share of the rows.
pub(crate) fn contiguous_groups(number_of_rows: usize, number_of_groups: usize) -> Array1<i64> {
    Array1::from_shape_fn(number_of_rows, |row| {
        ((row * number_of_groups) / number_of_rows) as i64
    })
}

/// Converts a dense design into CSR, storing only the non-zero entries.
pub(crate) fn sparse_design(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new(dense.dim());
    for ((row, column), value) in dense.indexed_iter() {
        if *value != 0.0 {
            triplets.add_triplet(row, column, *value);
        }
    }
    triplets.to_csr()
}
