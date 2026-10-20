//! Shared fixtures of the stacking tests.

use ndarray::Array2;
use sprs::{CsMat, TriMat};

use crate::simple_imputer::output::SKImputedOutput;

/// Build a CSR matrix from triplets for the sparse cases.
pub(super) fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

/// The dense payload of an imputation, panicking on a sparse one.
pub(super) fn dense_imputation(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => {
            panic!("expected a dense imputation, found a csr one: {matrix:?}")
        }
    }
}
