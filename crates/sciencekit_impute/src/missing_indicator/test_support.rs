//! Shared fixtures of the missing-indicator tests (spec `missing-indicator`).

use ndarray::Array2;
use sprs::{CsMat, TriMat};

use super::output::SKIndicatorOutput;

/// Build a CSR matrix from triplets for the sparse cases.
pub(super) fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

/// The dense payload of a mask, panicking on a sparse one.
pub(super) fn dense_mask(output: SKIndicatorOutput<f64>) -> Array2<f64> {
    match output {
        SKIndicatorOutput::Dense(matrix) => matrix,
        SKIndicatorOutput::Sparse(matrix) => {
            panic!("expected a dense mask, found a csr one: {matrix:?}")
        }
    }
}
