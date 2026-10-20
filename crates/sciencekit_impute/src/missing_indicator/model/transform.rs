//! The mask transform: dense marking, the CSR mask and the treatment of
//! missingness fit never saw (spec `missing-indicator`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{
    SKDataView, SKError, SKFeatureTransformer, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};
use sprs::{CsMat, CsMatView, TriMat};

use crate::missing_indicator::model::SKMissingIndicatorModel;
use crate::missing_indicator::output::SKIndicatorOutput;
use crate::missingness::{sk_is_missing, sk_sparse_marks_implicit_zeros};

impl<F: SKFloat> SKFeatureTransformer<F> for SKMissingIndicatorModel<F> {
    type Output = SKIndicatorOutput<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        match features.try_into()? {
            SKDataView::Dense(dense) => {
                self.validate_columns(dense.nrows(), dense.ncols())?;
                let attributes = self.transform_attributes(dense.nrows(), dense.ncols());
                sk_run_operation(attributes, || {
                    self.reject_new_missingness_dense(&dense)?;
                    if self.sparse_output {
                        Ok(SKIndicatorOutput::Sparse(
                            self.sparse_mask_from_dense(&dense),
                        ))
                    } else {
                        Ok(SKIndicatorOutput::Dense(self.dense_mask(&dense)))
                    }
                })
            }
            SKDataView::Sparse(sparse) => {
                self.validate_columns(sparse.rows(), sparse.cols())?;
                let attributes = self.transform_attributes(sparse.rows(), sparse.cols());
                sk_run_operation(attributes, || {
                    self.reject_new_missingness_sparse(&sparse)?;
                    Ok(SKIndicatorOutput::Sparse(self.sparse_mask(&sparse)))
                })
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the input to a dense or CSR matrix",
            }),
        }
    }
}

impl<F: SKFloat> SKMissingIndicatorModel<F> {
    /// Reject a column count that differs from the fitted one before any
    /// element is read.
    fn validate_columns(&self, rows_count: usize, columns_count: usize) -> Result<(), SKError> {
        if columns_count == self.number_of_input_columns {
            return Ok(());
        }
        Err(SKError::shape_mismatch_2d(
            rows_count,
            self.number_of_input_columns,
            rows_count,
            columns_count,
        ))
    }

    /// The span attributes of a transform, carrying the fitted execution mode
    /// and backend kind.
    fn transform_attributes(
        &self,
        rows_count: usize,
        columns_count: usize,
    ) -> SKOperationAttributes {
        SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        }
    }

    /// A dense mask, one column per feature in the fitted scope.
    fn dense_mask(&self, features: &ArrayView2<'_, F>) -> Array2<F> {
        Array2::from_shape_fn(
            (features.nrows(), self.scope.len()),
            |(row, mask_column)| {
                let value = features[(row, self.scope[mask_column])];
                if sk_is_missing(value, self.missing_values) {
                    F::one()
                } else {
                    F::zero()
                }
            },
        )
    }

    /// The same mask in CSR form, storing exactly the missing cells.
    fn sparse_mask_from_dense(&self, features: &ArrayView2<'_, F>) -> CsMat<F> {
        let mut triplets = TriMat::new((features.nrows(), self.scope.len()));
        for (mask_column, &column) in self.scope.iter().enumerate() {
            features
                .column(column)
                .iter()
                .enumerate()
                .filter(|(_, value)| sk_is_missing(**value, self.missing_values))
                .for_each(|(row, _)| triplets.add_triplet(row, mask_column, F::one()));
        }
        triplets.to_csr()
    }

    /// A sparse mask: the explicit missing entries plus the structural zeros
    /// when the marker claims them.
    fn sparse_mask(&self, features: &CsMatView<'_, F>) -> CsMat<F> {
        let mut triplets = TriMat::new((features.rows(), self.scope.len()));
        let marks_implicit_zeros = sk_sparse_marks_implicit_zeros(self.missing_values);
        for (row_index, row) in features.outer_iterator().enumerate() {
            for (mask_column, &column) in self.scope.iter().enumerate() {
                match row.get(column) {
                    Some(&value) if sk_is_missing(value, self.missing_values) => {
                        triplets.add_triplet(row_index, mask_column, F::one());
                    }
                    None if marks_implicit_zeros => {
                        triplets.add_triplet(row_index, mask_column, F::one());
                    }
                    _ => {}
                }
            }
        }
        triplets.to_csr()
    }

    /// Fail on a missing cell in a dense feature that was complete at fit
    /// time, naming the first offending column.
    fn reject_new_missingness_dense(&self, features: &ArrayView2<'_, F>) -> Result<(), SKError> {
        if !self.error_on_new {
            return Ok(());
        }
        let in_scope = self.scope_membership();
        for (column, values) in features.columns().into_iter().enumerate() {
            if in_scope[column] {
                continue;
            }
            if values
                .iter()
                .any(|&value| sk_is_missing(value, self.missing_values))
            {
                return Err(SKError::UnexpectedMissingness { column });
            }
        }
        Ok(())
    }

    /// Fail on a missing cell in a sparse feature that was complete at fit
    /// time, counting the implicit zeros the marker claims.
    fn reject_new_missingness_sparse(&self, features: &CsMatView<'_, F>) -> Result<(), SKError> {
        if !self.error_on_new {
            return Ok(());
        }
        let in_scope = self.scope_membership();
        let mut explicit_counts = vec![0usize; self.number_of_input_columns];
        for row in features.outer_iterator() {
            for (column, &value) in row.iter() {
                explicit_counts[column] += 1;
                if in_scope[column] {
                    continue;
                }
                if sk_is_missing(value, self.missing_values) {
                    return Err(SKError::UnexpectedMissingness { column });
                }
            }
        }
        if sk_sparse_marks_implicit_zeros(self.missing_values) {
            for (column, count) in explicit_counts.iter().enumerate() {
                if !in_scope[column] && *count < features.rows() {
                    return Err(SKError::UnexpectedMissingness { column });
                }
            }
        }
        Ok(())
    }

    /// A membership table of the fitted scope, indexed by input column.
    fn scope_membership(&self) -> Vec<bool> {
        let mut in_scope = vec![false; self.number_of_input_columns];
        for &column in &self.scope {
            in_scope[column] = true;
        }
        in_scope
    }
}
