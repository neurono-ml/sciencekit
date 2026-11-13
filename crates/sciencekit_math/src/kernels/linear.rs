//! Linear inner-product kernel `K(X, Y) = X · Yᵀ`
//! (spec `kernel-functions`).

use ndarray::{Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::{CsMatView, CsVecView};

use super::parallel::sk_fill_kernel_rows;
use super::validation::sk_validate_kernel_pair;
use crate::pairwise::{dot_product_simd, sk_sparse_dot_product};

/// Compute a single dense row dot product, using SIMD when both rows are
/// contiguous slices and an iterator fold when either side is strided.
pub(crate) fn sk_dense_row_dot<F: SKFloat>(left: ArrayView1<'_, F>, right: ArrayView1<'_, F>) -> F {
    if left.as_slice().is_some() && right.as_slice().is_some() {
        dot_product_simd(&left, &right)
    } else {
        left.iter()
            .zip(right.iter())
            .fold(F::zero(), |accumulator, (&a, &b)| accumulator + a * b)
    }
}

/// Compute the dot product between one CSR row and one dense row without
/// densifying the sparse operand.
pub(crate) fn sk_sparse_dense_row_dot<F: SKFloat>(
    sparse_row: CsVecView<'_, F>,
    dense_row: ArrayView1<'_, F>,
) -> F {
    sparse_row
        .indices()
        .iter()
        .zip(sparse_row.data().iter())
        .fold(F::zero(), |accumulator, (&column, &value)| {
            accumulator + value * dense_row[column]
        })
}

fn sk_dense_linear_kernel<F: SKFloat>(
    left: &ArrayView2<'_, F>,
    right: &ArrayView2<'_, F>,
    allow_parallel: bool,
) -> Array2<F> {
    let right_rows: Vec<_> = right.rows().into_iter().collect();
    sk_fill_kernel_rows(
        left.nrows(),
        right.nrows(),
        allow_parallel,
        |row_index, mut output_row| {
            let left_row = left.row(row_index);
            output_row
                .iter_mut()
                .zip(right_rows.iter())
                .for_each(|(cell, right_row)| {
                    *cell = sk_dense_row_dot(left_row, *right_row);
                });
        },
    )
}

fn sk_sparse_linear_kernel<F: SKFloat>(
    left: &CsMatView<'_, F>,
    right: &CsMatView<'_, F>,
    allow_parallel: bool,
) -> Array2<F> {
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    sk_fill_kernel_rows(
        left.rows(),
        right.rows(),
        allow_parallel,
        |row_index, mut output_row| {
            let left_row = left_rows[row_index];
            output_row
                .iter_mut()
                .zip(right_rows.iter())
                .for_each(|(cell, right_row)| {
                    *cell = sk_sparse_dot_product(
                        left_row.indices(),
                        left_row.data(),
                        right_row.indices(),
                        right_row.data(),
                    );
                });
        },
    )
}

fn sk_sparse_dense_linear_kernel<F: SKFloat>(
    left: &CsMatView<'_, F>,
    right: &ArrayView2<'_, F>,
    allow_parallel: bool,
) -> Array2<F> {
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.rows().into_iter().collect();
    sk_fill_kernel_rows(
        left.rows(),
        right.nrows(),
        allow_parallel,
        |row_index, mut output_row| {
            let left_row = left_rows[row_index];
            output_row
                .iter_mut()
                .zip(right_rows.iter())
                .for_each(|(cell, right_row)| {
                    *cell = sk_sparse_dense_row_dot(left_row, *right_row);
                });
        },
    )
}

fn sk_dense_sparse_linear_kernel<F: SKFloat>(
    left: &ArrayView2<'_, F>,
    right: &CsMatView<'_, F>,
    allow_parallel: bool,
) -> Array2<F> {
    let right_rows: Vec<_> = right.outer_iterator().collect();
    sk_fill_kernel_rows(
        left.nrows(),
        right.rows(),
        allow_parallel,
        |row_index, mut output_row| {
            let left_row = left.row(row_index);
            output_row
                .iter_mut()
                .zip(right_rows.iter())
                .for_each(|(cell, right_row)| {
                    *cell = sk_sparse_dense_row_dot(*right_row, left_row);
                });
        },
    )
}

/// Internal entry point allowing sequential or parallel row evaluation.
pub(crate) fn sk_linear_kernel_with_mode<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
    allow_parallel: bool,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_pair(&left, &right)?;
    match (left, right) {
        (SKDataView::Dense(left_dense), SKDataView::Dense(right_dense)) => Ok(
            sk_dense_linear_kernel(&left_dense, &right_dense, allow_parallel),
        ),
        (SKDataView::Sparse(left_sparse), SKDataView::Sparse(right_sparse)) => Ok(
            sk_sparse_linear_kernel(&left_sparse, &right_sparse, allow_parallel),
        ),
        (SKDataView::Sparse(left_sparse), SKDataView::Dense(right_dense)) => Ok(
            sk_sparse_dense_linear_kernel(&left_sparse, &right_dense, allow_parallel),
        ),
        (SKDataView::Dense(left_dense), SKDataView::Sparse(right_sparse)) => Ok(
            sk_dense_sparse_linear_kernel(&left_dense, &right_sparse, allow_parallel),
        ),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: left.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Compute the linear kernel matrix `K[i, j] = ⟨left[i], right[j]⟩`.
pub fn sk_linear_kernel_matrix<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    sk_linear_kernel_with_mode(left, right, true)
}
