//! Linear inner-product kernel `K(X, Y) = X · Yᵀ`
//! (spec `kernel-functions`).

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::validation::sk_validate_kernel_pair;
use crate::pairwise::dot_product_simd;

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

/// Dense×dense linear kernel matrix filled row by row.
fn sk_dense_linear_kernel<F: SKFloat>(left: &ArrayView2<'_, F>, right: &ArrayView2<'_, F>) -> Array2<F> {
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

/// Compute the linear kernel matrix `K[i, j] = ⟨left[i], right[j]⟩`.
pub fn sk_linear_kernel_matrix<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_pair(&left, &right)?;
    match (left, right) {
        (SKDataView::Dense(left_dense), SKDataView::Dense(right_dense)) => {
            Ok(sk_dense_linear_kernel(&left_dense, &right_dense))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: left.representation(),
            suggestion: "sparse linear kernel arrives in task 1.3",
        }),
    }
}
