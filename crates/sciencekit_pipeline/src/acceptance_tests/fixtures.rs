//! Deterministic datasets shared by the acceptance tests.

use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};

pub(super) fn tiny_features() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0,
        ],
    )
    .unwrap()
}

pub(super) fn tiny_labels() -> Array1<i64> {
    Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1])
}

pub(super) fn tiny_regression_targets() -> Array1<f64> {
    Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
}

/// A sparse twin of [`tiny_features`] where the zero cells are implicit.
pub(super) fn tiny_sparse_features() -> CsMat<f64> {
    let mut triplets = TriMat::new((8, 2));
    for (row, value) in tiny_features().rows().into_iter().enumerate() {
        for (column, &cell) in value.iter().enumerate() {
            if cell != 0.0 {
                triplets.add_triplet(row, column, cell);
            }
        }
    }
    triplets.to_csr()
}

/// A deterministic dense dataset of `rows × columns` values in `[0, 1)`.
pub(super) fn large_features(rows: usize, columns: usize) -> Array2<f64> {
    let values: Vec<f64> = (0..rows * columns)
        .map(|index| ((index * 7 + 3) % 101) as f64 / 101.0)
        .collect();
    Array2::from_shape_vec((rows, columns), values).unwrap()
}

pub(super) fn large_labels(rows: usize) -> Array1<i64> {
    Array1::from_vec((0..rows).map(|row| (row % 2) as i64).collect())
}
