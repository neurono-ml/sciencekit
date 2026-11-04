//! Representation transitions for the retained-design coordinate-descent state.
//!
//! A stream may start dense and later carry sparse blocks; the accumulator is
//! upgraded to CSR once and every later dense batch is converted, so no block is
//! ever densified.

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;
use sprs::{CsMat, TriMat};

use super::accumulator::SKCoordinateDescentState;

impl<F: SKFloat> SKCoordinateDescentState<F> {
    /// Convert a dense accumulator to the CSR variant, keeping its targets.
    pub(crate) fn ensure_sparse(&mut self) {
        if matches!(self, Self::Sparse { .. }) {
            return;
        }
        let previous = std::mem::replace(self, Self::new_sparse());
        if let Self::Dense { batches, targets } = previous {
            let columns_count = batches.first().map_or(0, ndarray::Array2::ncols);
            let sparse_batches = batches
                .iter()
                .map(|batch| sk_dense_to_csr(&batch.view()))
                .collect();
            *self = Self::Sparse {
                batches: sparse_batches,
                targets,
                columns_count,
            };
        }
    }

    /// Append one dense batch to an accumulator already in its sparse form.
    pub(crate) fn accumulate_dense_into_sparse(
        &mut self,
        features: ArrayView2<'_, F>,
        targets: ArrayView1<'_, F>,
    ) {
        let sparse = sk_dense_to_csr(&features);
        self.accumulate_sparse(sparse.view(), targets);
    }
}

/// Convert a dense design to CSR, dropping exact zeros.
pub(crate) fn sk_dense_to_csr<F: SKFloat>(features: &ArrayView2<'_, F>) -> CsMat<F> {
    let mut triplets = TriMat::new((features.nrows(), features.ncols()));
    features.indexed_iter().for_each(|((row, column), &value)| {
        if value != F::zero() {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}
