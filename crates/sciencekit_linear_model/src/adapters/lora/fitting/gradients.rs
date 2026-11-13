//! Representation-agnostic matrix products used by the adapter optimizers.

use ndarray::{Array2, ArrayView2, azip};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use crate::logistic_regression::validation::unsupported_representation;

/// `Xᵀ · gradient`, shaped `features × heads`, without densifying a sparse `X`.
pub(super) fn sk_transposed_feature_product<F: SKFloat>(
    data: &SKDataView<'_, F>,
    gradient: ArrayView2<'_, F>,
) -> Result<Array2<F>, SKError> {
    match data {
        SKDataView::Dense(dense) => Ok(dense.t().dot(&gradient)),
        SKDataView::Sparse(sparse) => {
            let mut product = Array2::<F>::zeros((sparse.cols(), gradient.ncols()));
            for (row_index, sparse_row) in sparse.outer_iterator().enumerate() {
                let gradient_row = gradient.row(row_index);
                for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
                    let mut output_row = product.row_mut(column);
                    azip!((accumulator in &mut output_row, &term in &gradient_row) {
                        *accumulator = *accumulator + value * term;
                    });
                }
            }
            Ok(product)
        }
        _ => Err(unsupported_representation()),
    }
}
