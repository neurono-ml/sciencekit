//! Dense and sparse feature row gathering.

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{SKDataView, SKError, SKFeatureBlock, SKFloat};
use sprs::{CsMat, CsMatView};

/// Gathers the selected rows of a feature block, preserving representation.
pub(crate) fn sk_gather_feature_rows<F: SKFloat>(
    features: SKDataView<'_, F>,
    row_indices: &[usize],
) -> Result<SKFeatureBlock<F>, SKError> {
    match features {
        SKDataView::Dense(view) => Ok(SKFeatureBlock::Dense(sk_gather_dense(view, row_indices)?)),
        SKDataView::Sparse(view) => {
            Ok(SKFeatureBlock::Sparse(sk_gather_sparse(view, row_indices)?))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the matrix to dense or CSR",
        }),
    }
}

fn sk_gather_dense<F: SKFloat>(
    view: ArrayView2<'_, F>,
    row_indices: &[usize],
) -> Result<Array2<F>, SKError> {
    for &row in row_indices {
        if row >= view.nrows() {
            return Err(SKError::InvalidHyperparameter {
                name: "row_indices",
                reason: format!(
                    "row index {row} is outside the {} rows of the input",
                    view.nrows()
                ),
            });
        }
    }
    Ok(view.select(Axis(0), row_indices))
}

fn sk_gather_sparse<F: SKFloat>(
    view: CsMatView<'_, F>,
    row_indices: &[usize],
) -> Result<CsMat<F>, SKError> {
    let mut indptr = Vec::with_capacity(row_indices.len() + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);

    for &row in row_indices {
        if row >= view.rows() {
            return Err(SKError::InvalidHyperparameter {
                name: "row_indices",
                reason: format!(
                    "row index {row} is outside the {} rows of the input",
                    view.rows()
                ),
            });
        }
        let outer = view
            .outer_view(row)
            .expect("a validated row index has an outer view");
        indices.extend_from_slice(outer.indices());
        data.extend(outer.data().iter().copied());
        indptr.push(data.len());
    }

    Ok(CsMat::new(
        (row_indices.len(), view.cols()),
        indptr,
        indices,
        data,
    ))
}
