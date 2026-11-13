//! The owned training design a fitted neighbor model stores
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::{CsMat, CsMatView, TriMat};

/// The training design owned by a fitted neighbor model.
///
/// Prediction must outlive the caller's buffers, so the design is copied once
/// at fit time. The representation is preserved: a sparse training set stays
/// CSR and is never densified.
#[derive(Debug, Clone)]
pub enum SKTrainingData<F> {
    /// A row-major dense feature matrix.
    Dense(Array2<F>),
    /// A compressed sparse row feature matrix.
    Sparse(CsMat<F>),
}

impl<F: SKFloat> SKTrainingData<F> {
    /// Copy a borrowed view into the owned form, normalizing sparse storage to
    /// CSR without densifying it.
    pub fn from_view(view: SKDataView<'_, F>) -> Self {
        match view {
            SKDataView::Dense(dense) => SKTrainingData::Dense(dense.to_owned()),
            SKDataView::Sparse(sparse) => SKTrainingData::Sparse(sk_to_csr(sparse)),
            _ => SKTrainingData::Dense(Array2::zeros((0, 0))),
        }
    }

    /// Borrow the design back as a zero-copy view.
    pub fn view(&self) -> SKDataView<'_, F> {
        match self {
            SKTrainingData::Dense(dense) => SKDataView::Dense(dense.view()),
            SKTrainingData::Sparse(sparse) => SKDataView::Sparse(sparse.view()),
        }
    }

    /// The number of training rows.
    pub fn rows(&self) -> usize {
        match self {
            SKTrainingData::Dense(dense) => dense.nrows(),
            SKTrainingData::Sparse(sparse) => sparse.rows(),
        }
    }

    /// The number of features per training row.
    pub fn columns(&self) -> usize {
        match self {
            SKTrainingData::Dense(dense) => dense.ncols(),
            SKTrainingData::Sparse(sparse) => sparse.cols(),
        }
    }

    /// The representation label (`dense` or `csr`).
    pub fn representation(&self) -> &'static str {
        match self {
            SKTrainingData::Dense(_) => "dense",
            SKTrainingData::Sparse(_) => "csr",
        }
    }

    /// Whether the design is the dense representation.
    pub fn is_dense(&self) -> bool {
        matches!(self, SKTrainingData::Dense(_))
    }

    /// Borrow the dense matrix when the design is dense.
    pub fn dense(&self) -> Option<ArrayView2<'_, F>> {
        match self {
            SKTrainingData::Dense(dense) => Some(dense.view()),
            SKTrainingData::Sparse(_) => None,
        }
    }

    /// Reject any NaN or infinity, naming its position in either
    /// representation.
    pub fn validate_finite(&self) -> Result<(), SKError> {
        match self {
            SKTrainingData::Dense(dense) => {
                if let Some(position) = dense.iter().position(|value| !value.is_finite()) {
                    return Err(SKError::NonFiniteValue { position });
                }
                Ok(())
            }
            SKTrainingData::Sparse(sparse) => {
                let columns = sparse.cols();
                for (row, stored) in sparse.outer_iterator().enumerate() {
                    for (column, value) in stored.iter() {
                        if !value.is_finite() {
                            return Err(SKError::NonFiniteValue {
                                position: row * columns + column,
                            });
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

/// Copy a borrowed sparse view into an owned CSR matrix.
///
/// A CSR input is copied verbatim; a CSC input is re-coordinated through
/// triplets, never densified.
fn sk_to_csr<F: SKFloat>(sparse: CsMatView<'_, F>) -> CsMat<F> {
    if sparse.storage() == sprs::CompressedStorage::CSR {
        return sparse.to_owned();
    }
    let mut triplet = TriMat::new((sparse.rows(), sparse.cols()));
    for (value, (row, column)) in sparse.iter() {
        triplet.add_triplet(row, column, *value);
    }
    triplet.to_csr()
}
