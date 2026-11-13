//! Structure-only split criterion and input validation helpers.

use sciencekit_common::{SKDataView, SKError, SKFloat};

use crate::criterion::SKTreeCriterion as SKConfiguredCriterion;
use crate::growth::SKTreeCriterion;
use crate::impurity::sk_squared_error_impurity;
use crate::splitter::SKScanContext;

/// Structure-only squared-error split criterion whose leaf output is a neutral
/// zero placeholder until the caller supplies leaf values.
pub(super) struct SKStructureCriterion<'a, F> {
    pub(super) pseudo_residuals: &'a [F],
}

impl<F: SKFloat> SKTreeCriterion<F> for SKStructureCriterion<'_, F> {
    fn impurity(&self, sample_indices: &[usize]) -> F {
        let values: Vec<F> = sample_indices
            .iter()
            .map(|&row| self.pseudo_residuals[row])
            .collect();
        sk_squared_error_impurity(&values)
    }

    fn leaf_output(&self, _sample_indices: &[usize]) -> Vec<F> {
        vec![F::zero()]
    }

    fn scan_context(&self) -> SKScanContext<'_, F> {
        SKScanContext::Regression {
            responses: self.pseudo_residuals,
            criterion: SKConfiguredCriterion::SquaredError,
        }
    }
}

pub(super) fn sk_structure_dimensions<F>(
    features: &SKDataView<'_, F>,
) -> Result<(usize, usize), SKError> {
    let (rows, columns) = match features {
        SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
        SKDataView::Sparse(sparse) if sparse.is_csr() => (sparse.rows(), sparse.cols()),
        SKDataView::Sparse(_) => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "csc",
                suggestion: "convert the sparse matrix to CSR with to_csr",
            });
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            });
        }
    };
    if rows == 0 {
        return Err(SKError::EmptyInput {
            operation: "build_structure",
        });
    }
    if columns == 0 {
        return Err(SKError::shape_mismatch_2d(rows, 1, rows, 0));
    }
    Ok((rows, columns))
}
