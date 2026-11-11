//! Single-dispatch dense/CSR right multiplication (`left · right_factor`)
//! without sparse densification (`spec` `truncated-svd`, decision 2).

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use sciencekit_common::{SKBackendKind, SKDataView, SKError};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::sparse_ops::sk_csr_dense_product;

pub(super) const ROW_BLOCK_SIZE: usize = 256;

#[cfg(test)]
static PEAK_MATERIALIZED_ELEMENTS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn sk_reset_peak_materialized_elements() {
    PEAK_MATERIALIZED_ELEMENTS.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(super) fn sk_peak_materialized_elements() -> usize {
    PEAK_MATERIALIZED_ELEMENTS.load(Ordering::SeqCst)
}

#[cfg(test)]
fn sk_record_materialized_elements(elements: usize) {
    PEAK_MATERIALIZED_ELEMENTS.fetch_max(elements, Ordering::SeqCst);
}

/// Computes `left · right_factor` with a single representation dispatch per
/// call, never materializing a dense copy of a sparse CSR input.
pub(super) fn sk_multiply_view_by_dense<F: SKBackendSelection>(
    left: SKDataView<'_, F>,
    right_factor: ArrayView2<'_, F>,
    backend_kind: SKBackendKind,
    parallelism: usize,
) -> Result<Array2<F>, SKError> {
    match left {
        SKDataView::Dense(dense) => {
            let rows_count = dense.nrows();
            let backend = sk_math_backend_for_kind::<F>(backend_kind);
            let mut output = Array2::<F>::zeros((rows_count, right_factor.ncols()));
            if parallelism > 1 && rows_count > ROW_BLOCK_SIZE {
                dense
                    .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
                    .into_par_iter()
                    .zip(
                        output
                            .axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE)
                            .into_par_iter(),
                    )
                    .for_each(|(input_chunk, mut output_chunk)| {
                        let block = backend.gemm(input_chunk, right_factor, F::one(), 1);
                        output_chunk.assign(&block);
                    });
            } else {
                for (input_chunk, mut output_chunk) in dense
                    .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
                    .zip(output.axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE))
                {
                    let block = backend.gemm(input_chunk, right_factor, F::one(), 1);
                    output_chunk.assign(&block);
                }
            }
            #[cfg(test)]
            sk_record_materialized_elements(output.len());
            Ok(output)
        }
        SKDataView::Sparse(sparse) => {
            let output = sk_csr_dense_product(&sparse, &right_factor);
            #[cfg(test)]
            sk_record_materialized_elements(output.len());
            Ok(output)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: left.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}
