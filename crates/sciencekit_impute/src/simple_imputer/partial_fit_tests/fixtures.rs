//! Shared fixtures of the incremental-fit tests.

use ndarray::Array2;
use sprs::{CsMat, TriMat};

use crate::simple_imputer::model::SKSimpleImputerModel;

/// The documented relative tolerance of the online median sketch.
pub(super) const SKETCH_RELATIVE_TOLERANCE: f64 = 0.05;

/// Build a CSR matrix from triplets for the sparse cases.
pub(super) fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

/// The statistics as a plain vector.
pub(super) fn statistics(model: &SKSimpleImputerModel<f64>) -> Vec<f64> {
    model.statistics().to_vec()
}

/// Assert two statistic vectors agree within a relative tolerance.
pub(super) fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (column, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let difference = (actual - expected).abs();
        let scale = expected.abs().max(1.0);
        assert!(
            difference <= tolerance * scale,
            "column {column}: {actual} differs from {expected} beyond {tolerance}"
        );
    }
}

/// A deterministic column of values with every seventh cell missing.
pub(super) fn median_features(rows_count: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows_count, 1), |(row, _)| {
        if row % 7 == 0 {
            f64::NAN
        } else {
            ((row * 37) % rows_count) as f64 + 1.0
        }
    })
}
