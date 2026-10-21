//! Shared sparse test helpers: dense-to-CSR conversion and a converging
//! `SparseCg` estimator.

use ndarray::Array2;
use sciencekit_common::builders::SKBuilder;
use sprs::{CsMat, TriMat};

use super::builder::SKRidgeSolver;
use super::core_implementation::SKRidge;

/// The relative tolerance the documented sparse-versus-dense agreement holds to.
pub(super) const SPARSE_AGREEMENT_TOLERANCE: f64 = 1e-6;

/// Convert a dense design to an equivalent CSR matrix, dropping exact zeros.
pub(super) fn to_csr(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense.indexed_iter().for_each(|((row, column), &value)| {
        if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}

/// A converging `SparseCg` estimator over `f64`.
pub(super) fn sparse_cg(alpha: f64) -> SKRidge<f64> {
    SKRidge::build()
        .alpha(alpha)
        .solver(SKRidgeSolver::SparseCg)
        .tolerance(1e-12)
        .maximum_number_of_iterations(10_000)
        .build()
        .expect("sparse-cg ridge builds")
}
