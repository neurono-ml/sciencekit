//! Owned support-vector feature blocks and input validation
//! (spec `support-vector-machines`).

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::{CompressedStorage, CsMat, CsMatView};

/// Owned feature matrix retaining the training representation (dense or CSR)
/// without ever densifying sparse rows.
#[derive(Debug, Clone)]
pub enum SKOwnedFeatureBlock<F: SKFloat> {
    /// Row-major dense support-vector matrix.
    Dense(Array2<F>),
    /// Compressed-sparse-row support-vector matrix.
    Sparse(CsMat<F>),
}

impl<F: SKFloat> SKOwnedFeatureBlock<F> {
    /// Copy the selected `row_indices` out of `features`, preserving dense or
    /// CSR storage without densifying.
    pub fn select_rows(
        features: SKDataView<'_, F>,
        row_indices: &[usize],
    ) -> Result<Self, SKError> {
        match features {
            SKDataView::Dense(dense) => {
                let columns = dense.ncols();
                let mut selected = Array2::zeros((row_indices.len(), columns));
                row_indices
                    .iter()
                    .enumerate()
                    .for_each(|(target_row, &source_row)| {
                        selected.row_mut(target_row).assign(&dense.row(source_row));
                    });
                Ok(Self::Dense(selected))
            }
            SKDataView::Sparse(sparse) => {
                if sparse.storage() != CompressedStorage::CSR {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: "csc",
                        suggestion: "convert the sparse matrix to CSR",
                    });
                }
                Ok(Self::Sparse(sk_select_csr_rows(sparse, row_indices)))
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a dense matrix or a CSR matrix",
            }),
        }
    }

    /// Borrow the owned block back as an [`SKDataView`].
    pub fn view(&self) -> SKDataView<'_, F> {
        match self {
            Self::Dense(dense) => SKDataView::Dense(dense.view()),
            Self::Sparse(sparse) => SKDataView::Sparse(sparse.view()),
        }
    }

    /// Number of stored rows (support vectors).
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.nrows(),
            Self::Sparse(sparse) => sparse.rows(),
        }
    }

    /// Number of feature columns.
    pub fn columns(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ncols(),
            Self::Sparse(sparse) => sparse.cols(),
        }
    }

    /// Whether the stored block uses CSR sparse storage.
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }
}

fn sk_select_csr_rows<F: SKFloat>(sparse: CsMatView<'_, F>, row_indices: &[usize]) -> CsMat<F> {
    let columns = sparse.cols();
    let indptr_raw = sparse.indptr();
    let indices_raw = sparse.indices();
    let data_raw = sparse.data();

    let mut indptr = Vec::with_capacity(row_indices.len() + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);

    for &row in row_indices {
        let range = indptr_raw.outer_inds_sz(row);
        indices.extend_from_slice(&indices_raw[range.clone()]);
        data.extend_from_slice(&data_raw[range]);
        indptr.push(indices.len());
    }
    CsMat::new((row_indices.len(), columns), indptr, indices, data)
}
