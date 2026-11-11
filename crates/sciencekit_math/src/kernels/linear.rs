//! Linear inner-product kernel `K(X, Y) = X · Yᵀ`
//! (spec `kernel-functions`).

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::{CsMatView, CsVecView};

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

/// Dense×dense linear kernel matrix filled sequentially row by row.
pub(crate) fn sk_dense_linear_kernel_sequential<F: SKFloat>(
    left: &ArrayView2<'_, F>,
    right: &ArrayView2<'_, F>,
) -> Array2<F> {
    let mut output = Array2::zeros((left.nrows(), right.nrows()));
    let right_rows: Vec<_> = right.rows().into_iter().collect();
    output
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(row_index, mut output_row)| {
            let left_row = left.row(row_index);
            output_row
                .iter_mut()
                .zip(right_rows.iter())
                .for_each(|(cell, right_row)| {
                    *cell = sk_dense_row_dot(left_row, *right_row);
                });
        });
    output
}

/// Sparse×sparse linear kernel matrix filled sequentially row by row.
pub(crate) fn sk_sparse_linear_kernel_sequential<F: SKFloat>(
    left: &CsMatView<'_, F>,
    right: &CsMatView<'_, F>,
) -> Array2<F> {
    let mut output = Array2::zeros((left.rows(), right.rows()));
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.outer_iterator().collect();
    output
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(row_index, mut output_row)| {
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
        });
    output
}

/// Sparse×dense linear kernel matrix filled sequentially row by row.
pub(crate) fn sk_sparse_dense_linear_kernel_sequential<F: SKFloat>(
    left: &CsMatView<'_, F>,
    right: &ArrayView2<'_, F>,
) -> Array2<F> {
    let mut output = Array2::zeros((left.rows(), right.nrows()));
    let left_rows: Vec<_> = left.outer_iterator().collect();
    let right_rows: Vec<_> = right.rows().into_iter().collect();
    output
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(row_index, mut output_row)| {
            let left_row = left_rows[row_index];
            output_row
                .iter_mut()
                .zip(right_rows.iter())
                .for_each(|(cell, right_row)| {
                    *cell = sk_sparse_dense_row_dot(left_row, *right_row);
                });
        });
    output
}

/// Dense×sparse linear kernel matrix filled sequentially row by row.
pub(crate) fn sk_dense_sparse_linear_kernel_sequential<F: SKFloat>(
    left: &ArrayView2<'_, F>,
    right: &CsMatView<'_, F>,
) -> Array2<F> {
    let mut output = Array2::zeros((left.nrows(), right.rows()));
    let right_rows: Vec<_> = right.outer_iterator().collect();
    output
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(row_index, mut output_row)| {
            let left_row = left.row(row_index);
            output_row
                .iter_mut()
                .zip(right_rows.iter())
                .for_each(|(cell, right_row)| {
                    *cell = sk_sparse_dense_row_dot(*right_row, left_row);
                });
        });
    output
}

/// Compute the linear kernel matrix `K[i, j] = ⟨left[i], right[j]⟩`.
///
/// Dispatches on the input representations once and never densifies a sparse
/// operand.
pub fn sk_linear_kernel_matrix<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_pair(&left, &right)?;
    match (left, right) {
        (SKDataView::Dense(left_dense), SKDataView::Dense(right_dense)) => {
            Ok(sk_dense_linear_kernel_sequential(&left_dense, &right_dense))
        }
        (SKDataView::Sparse(left_sparse), SKDataView::Sparse(right_sparse)) => Ok(
            sk_sparse_linear_kernel_sequential(&left_sparse, &right_sparse),
        ),
        (SKDataView::Sparse(left_sparse), SKDataView::Dense(right_dense)) => Ok(
            sk_sparse_dense_linear_kernel_sequential(&left_sparse, &right_dense),
        ),
        (SKDataView::Dense(left_dense), SKDataView::Sparse(right_sparse)) => Ok(
            sk_dense_sparse_linear_kernel_sequential(&left_dense, &right_sparse),
        ),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: left.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}
