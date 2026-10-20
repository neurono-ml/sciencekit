//! Horizontal stacking of the imputed features and the missingness mask
//! (spec `simple-imputer`).
//!
//! The mask always comes after the imputed features, so the output width is
//! the retained feature count plus the mask width.

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{SKError, SKFloat};
use sprs::{CsMat, TriMat};

/// Concatenate the imputed features and the mask along the column axis.
pub(super) fn horizontally_stack_dense<F: SKFloat>(
    imputed: &Array2<F>,
    mask: &ArrayView2<'_, F>,
) -> Result<Array2<F>, SKError> {
    ndarray::concatenate(Axis(1), &[imputed.view(), *mask]).map_err(|_| {
        SKError::Conversion(
            "the imputed features and the missingness mask must share their row count".into(),
        )
    })
}

/// Concatenate the imputed features and the mask along the column axis,
/// shifting the mask columns past the feature columns.
pub(super) fn horizontally_stack_sparse<F: SKFloat>(
    imputed: &CsMat<F>,
    mask: &CsMat<F>,
) -> CsMat<F> {
    let mut triplets = TriMat::new((imputed.rows(), imputed.cols() + mask.cols()));
    for (row_index, row) in imputed.outer_iterator().enumerate() {
        for (column, &value) in row.iter() {
            triplets.add_triplet(row_index, column, value);
        }
    }
    for (row_index, row) in mask.outer_iterator().enumerate() {
        for (column, &value) in row.iter() {
            triplets.add_triplet(row_index, imputed.cols() + column, value);
        }
    }
    triplets.to_csr()
}
