//! The fit design: dense or CSR plus its CSC re-indexing.

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};
use sprs::{CsMatView, CsVecView};

/// A feature design prepared for split search.
///
/// A dense design borrows the caller's matrix. A sparse design borrows the
/// caller's CSR view and a CSC re-indexing built once at fit time; the CSC copy
/// is a re-indexing of the stored entries, never a densification.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SKDesign<'a, F> {
    /// Row-major dense features.
    Dense(ArrayView2<'a, F>),
    /// CSR features plus their CSC re-indexing for column-wise scans.
    Sparse {
        /// The original CSR view, used for row-wise value lookup.
        csr: CsMatView<'a, F>,
        /// The column-wise re-indexing used by the split scan.
        csc: CsMatView<'a, F>,
    },
}

impl<'a, F: SKFloat> SKDesign<'a, F> {
    /// The number of rows.
    pub(crate) fn rows(&self) -> usize {
        match self {
            SKDesign::Dense(dense) => dense.nrows(),
            SKDesign::Sparse { csr, .. } => csr.rows(),
        }
    }

    /// The number of columns.
    pub(crate) fn columns(&self) -> usize {
        match self {
            SKDesign::Dense(dense) => dense.ncols(),
            SKDesign::Sparse { csr, .. } => csr.cols(),
        }
    }

    /// Whether the design is stored densely.
    #[cfg(test)]
    pub(crate) fn is_dense(&self) -> bool {
        matches!(self, SKDesign::Dense(_))
    }

    /// The value of one feature for one sample; an absent sparse entry is zero.
    pub(crate) fn feature_value(&self, row: usize, feature: usize) -> F {
        match self {
            SKDesign::Dense(dense) => dense[[row, feature]],
            SKDesign::Sparse { csr, .. } => {
                sk_sparse_row_value(&csr.outer_view(row).expect("row within bounds"), feature)
            }
        }
    }

    /// The node's samples ordered by one feature value, then sample index.
    ///
    /// Ties are broken by the sample index so the scan order is deterministic.
    pub(crate) fn sorted_feature_values(
        &self,
        sample_indices: &[usize],
        feature: usize,
    ) -> Vec<(F, usize)> {
        let mut values = match self {
            SKDesign::Dense(dense) => sample_indices
                .iter()
                .map(|&row| (dense[[row, feature]], row))
                .collect::<Vec<_>>(),
            SKDesign::Sparse { csc, .. } => {
                self.sparse_feature_values(sample_indices, csc, feature)
            }
        };
        values.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .expect("finite feature values")
                .then_with(|| left.1.cmp(&right.1))
        });
        values
    }

    /// Collect the node's stored entries column-wise and fill implicit zeros.
    fn sparse_feature_values(
        &self,
        sample_indices: &[usize],
        csc: &CsMatView<'a, F>,
        feature: usize,
    ) -> Vec<(F, usize)> {
        let mut ordered = sample_indices.to_vec();
        ordered.sort_unstable();
        let column = csc.outer_view(feature).expect("column within bounds");
        let mut values = Vec::with_capacity(ordered.len());
        let mut present = Vec::new();
        for (&row, &value) in column.indices().iter().zip(column.data().iter()) {
            if ordered.binary_search(&row).is_ok() {
                values.push((value, row));
                present.push(row);
            }
        }
        present.sort_unstable();
        let mut cursor = 0;
        for &row in &ordered {
            while cursor < present.len() && present[cursor] < row {
                cursor += 1;
            }
            if cursor == present.len() || present[cursor] != row {
                values.push((F::zero(), row));
            }
        }
        values
    }

    /// Partition the node's samples by the candidate threshold.
    pub(crate) fn partition(
        &self,
        sample_indices: &[usize],
        feature: usize,
        threshold: F,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &row in sample_indices {
            if self.feature_value(row, feature) <= threshold {
                left.push(row);
            } else {
                right.push(row);
            }
        }
        (left, right)
    }

    /// Reject any non-finite stored value, naming its flattened position.
    pub(crate) fn validate_finite(&self) -> Result<(), SKError> {
        let columns = self.columns();
        match self {
            SKDesign::Dense(dense) => {
                for (row, row_values) in dense.rows().into_iter().enumerate() {
                    for (column, &value) in row_values.iter().enumerate() {
                        if !value.is_finite() {
                            return Err(SKError::NonFiniteValue {
                                position: row * columns + column,
                            });
                        }
                    }
                }
            }
            SKDesign::Sparse { csr, .. } => {
                for (row, sparse_row) in csr.outer_iterator().enumerate() {
                    for (&column, &value) in
                        sparse_row.indices().iter().zip(sparse_row.data().iter())
                    {
                        if !value.is_finite() {
                            return Err(SKError::NonFiniteValue {
                                position: row * columns + column,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Read one stored feature value, treating an absent entry as zero.
pub(crate) fn sk_sparse_row_value<F: SKFloat>(row: &CsVecView<'_, F>, feature: usize) -> F {
    match row.indices().binary_search(&feature) {
        Ok(position) => row.data()[position],
        Err(_) => F::zero(),
    }
}
