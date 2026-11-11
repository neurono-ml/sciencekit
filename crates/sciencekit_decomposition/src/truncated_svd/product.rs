//! Single-dispatch dense/CSR right multiplication (`left · right_factor`)
//! without sparse densification (`spec` `truncated-svd`, decision 2).

#[cfg(test)]
use std::cell::Cell;

use ndarray::{Array2, ArrayView2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::{SKBackendKind, SKDataView, SKError};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::sparse_ops::sk_csr_dense_product;

pub(super) const ROW_BLOCK_SIZE: usize = 256;

#[cfg(test)]
thread_local! {
    static PEAK_MATERIALIZED_ELEMENTS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn sk_reset_peak_materialized_elements() {
    PEAK_MATERIALIZED_ELEMENTS.with(|peak| peak.set(0));
}

#[cfg(test)]
pub(super) fn sk_peak_materialized_elements() -> usize {
    PEAK_MATERIALIZED_ELEMENTS.with(Cell::get)
}

#[cfg(test)]
fn sk_record_materialized_elements(elements: usize) {
    PEAK_MATERIALIZED_ELEMENTS.with(|peak| peak.set(peak.get().max(elements)));
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
            let rows_count = sparse.rows();
            let output = if parallelism > 1 && rows_count > ROW_BLOCK_SIZE {
                let mut buffer = Array2::<F>::zeros((rows_count, right_factor.ncols()));
                buffer
                    .axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE)
                    .enumerate()
                    .par_bridge()
                    .for_each(|(chunk_index, mut output_chunk)| {
                        let start_row = chunk_index * ROW_BLOCK_SIZE;
                        for (offset, mut output_row) in
                            output_chunk.rows_mut().into_iter().enumerate()
                        {
                            if let Some(sparse_row) = sparse.outer_view(start_row + offset) {
                                for (column, &value) in sparse_row.iter() {
                                    let dense_row = right_factor.row(column);
                                    azip!((accumulator in &mut output_row, &factor in dense_row) {
                                        *accumulator = *accumulator + value * factor;
                                    });
                                }
                            }
                        }
                    });
                buffer
            } else {
                sk_csr_dense_product(&sparse, &right_factor)
            };
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
