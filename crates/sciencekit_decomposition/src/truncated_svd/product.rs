//! Non-densifying dense/CSR right multiplication (`left · right_factor`)
//! (`spec` `truncated-svd`, decision 2).

#[cfg(test)]
use std::cell::Cell;

use ndarray::{Array2, ArrayView2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::SKBackendKind;
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::sparse_ops::sk_csr_dense_product;

use super::validation::SKTruncatedInput;

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

/// Computes `left · right_factor` over the pre-dispatched dense or CSR view,
/// never materializing a dense copy of a sparse CSR input.
pub(super) fn sk_multiply_view_by_dense<F: SKBackendSelection>(
    left: SKTruncatedInput<'_, F>,
    right_factor: ArrayView2<'_, F>,
    backend_kind: SKBackendKind,
    parallelism: usize,
) -> Array2<F> {
    let output = match left {
        SKTruncatedInput::Dense(dense) => {
            let rows_count = dense.nrows();
            let backend = sk_math_backend_for_kind::<F>(backend_kind);
            let mut buffer = Array2::<F>::zeros((rows_count, right_factor.ncols()));
            if parallelism > 1 && rows_count > ROW_BLOCK_SIZE {
                dense
                    .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
                    .into_par_iter()
                    .zip(
                        buffer
                            .axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE)
                            .into_par_iter(),
                    )
                    .for_each(|(input_chunk, mut output_chunk)| {
                        let block = backend.gemm(input_chunk, right_factor, F::one(), 1);
                        output_chunk.assign(&block);
                    });
            } else {
                dense
                    .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
                    .zip(buffer.axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE))
                    .for_each(|(input_chunk, mut output_chunk)| {
                        let block = backend.gemm(input_chunk, right_factor, F::one(), 1);
                        output_chunk.assign(&block);
                    });
            }
            buffer
        }
        SKTruncatedInput::SparseCsr(sparse) => {
            let rows_count = sparse.rows();
            if parallelism > 1 && rows_count > ROW_BLOCK_SIZE {
                let mut buffer = Array2::<F>::zeros((rows_count, right_factor.ncols()));
                buffer
                    .axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE)
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(chunk_index, mut output_chunk)| {
                        let start_row = chunk_index * ROW_BLOCK_SIZE;
                        output_chunk
                            .rows_mut()
                            .into_iter()
                            .enumerate()
                            .for_each(|(offset, mut output_row)| {
                                if let Some(sparse_row) = sparse.outer_view(start_row + offset) {
                                    sparse_row.iter().for_each(|(column, &value)| {
                                        let dense_row = right_factor.row(column);
                                        azip!((accumulator in &mut output_row, &factor in dense_row) {
                                            *accumulator = *accumulator + value * factor;
                                        });
                                    });
                                }
                            });
                    });
                buffer
            } else {
                sk_csr_dense_product(&sparse, &right_factor)
            }
        }
    };
    #[cfg(test)]
    sk_record_materialized_elements(output.len());
    output
}
