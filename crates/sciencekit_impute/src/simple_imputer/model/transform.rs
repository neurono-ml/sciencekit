//! The imputation transform: dense fill and two-pass CSR rebuild
//! (spec `simple-imputer`).
//!
//! The transform changes only the cells the marker identifies as missing; an
//! observed cell is copied bit-identically. The output representation matches
//! the input, and a column the fit recorded as dropped is removed from both
//! representations.

use ndarray::{Array2, ArrayView2, azip};
use sciencekit_common::{
    SKDataView, SKError, SKFeatureTransformer, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};
use sprs::{CsMatView, TriMat};

use crate::missing_indicator::SKIndicatorOutput;
use crate::missingness::{sk_is_missing, sk_sparse_marks_implicit_zeros};
use crate::simple_imputer::model::SKSimpleImputerModel;
use crate::simple_imputer::model::stacking::{horizontally_stack_dense, horizontally_stack_sparse};
use crate::simple_imputer::output::SKImputedOutput;

impl<F: SKFloat> SKFeatureTransformer<F> for SKSimpleImputerModel<F> {
    type Output = SKImputedOutput<F>;
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
                    Ok(SKImputedOutput::Dense(self.transform_dense(&dense)?))
                })
            }
            SKDataView::Sparse(sparse) => {
                self.validate_columns(sparse.rows(), sparse.cols())?;
                let attributes = self.transform_attributes(sparse.rows(), sparse.cols());
                sk_run_operation(attributes, || {
                    Ok(SKImputedOutput::Sparse(self.transform_sparse(&sparse)?))
                })
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the input to a dense or CSR matrix",
            }),
        }
    }
}

impl<F: SKFloat> SKSimpleImputerModel<F> {
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

    /// The span attributes of a transform, carrying the carried execution mode
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

    /// The value that fills a missing cell of the column with `statistic`: the
    /// statistic itself, or `0.0` for a retained empty column (no statistic).
    fn fill_for(&self, statistic: F) -> F {
        if statistic.is_nan() {
            F::zero()
        } else {
            statistic
        }
    }

    /// Whether the source column at `column` was dropped by the fit.
    fn is_dropped(&self, column: usize) -> bool {
        self.dropped_columns.binary_search(&column).is_ok()
    }

    /// Fill the missing cells of every retained column, skipping the dropped
    /// ones, then stack the mask when `add_indicator` was requested.
    fn transform_dense(&self, features: &ArrayView2<'_, F>) -> Result<Array2<F>, SKError> {
        let mut output = Array2::zeros((features.nrows(), self.number_of_output_features()));
        let mut target = 0;
        for (source, &statistic) in self.statistics.iter().enumerate() {
            if self.is_dropped(source) {
                continue;
            }
            let fill = self.fill_for(statistic);
            let mut output_column = output.column_mut(target);
            let source_column = features.column(source);
            azip!((target_value in &mut output_column, &source_value in &source_column) {
                *target_value = if sk_is_missing(source_value, self.missing_values) {
                    fill
                } else {
                    source_value
                };
            });
            target += 1;
        }
        let Some(indicator) = self.indicator.as_ref() else {
            return Ok(output);
        };
        let mask = match indicator.transform(*features)? {
            SKIndicatorOutput::Dense(mask) => mask,
            SKIndicatorOutput::Sparse(mask) => mask.to_dense(),
        };
        horizontally_stack_dense(&output, &mask.view())
    }

    /// Rebuild the CSR matrix row by row: explicit entries are kept, missing
    /// ones become their column statistic, and — under a `0.0` marker — every
    /// implicit zero is filled too. The mask columns are stacked last when
    /// `add_indicator` was requested.
    fn transform_sparse(&self, features: &CsMatView<'_, F>) -> Result<sprs::CsMat<F>, SKError> {
        // The retained source columns in ascending order, with their output
        // index and the value that fills them.
        let mut retained: Vec<(usize, usize, F)> = Vec::new();
        for (source, &statistic) in self.statistics.iter().enumerate() {
            if !self.is_dropped(source) {
                retained.push((source, retained.len(), self.fill_for(statistic)));
            }
        }
        let fill_implicit_zeros = sk_sparse_marks_implicit_zeros(self.missing_values);
        let mut triplets = TriMat::new((features.rows(), self.number_of_output_features()));
        for (row_index, row) in features.outer_iterator().enumerate() {
            let mut cursor = 0;
            for (source, &value) in row.iter() {
                // Retained columns that this row leaves implicit.
                while cursor < retained.len() && retained[cursor].0 < source {
                    let (_, output, fill) = retained[cursor];
                    if fill_implicit_zeros && fill != F::zero() {
                        triplets.add_triplet(row_index, output, fill);
                    }
                    cursor += 1;
                }
                let Some(&(retained_source, output, fill)) = retained.get(cursor) else {
                    continue;
                };
                if retained_source != source {
                    continue;
                }
                cursor += 1;
                let imputed = if sk_is_missing(value, self.missing_values) {
                    fill
                } else {
                    value
                };
                if !(fill_implicit_zeros && imputed == F::zero()) {
                    triplets.add_triplet(row_index, output, imputed);
                }
            }
            // Retained columns past the last explicit entry of the row.
            while cursor < retained.len() {
                let (_, output, fill) = retained[cursor];
                if fill_implicit_zeros && fill != F::zero() {
                    triplets.add_triplet(row_index, output, fill);
                }
                cursor += 1;
            }
        }
        let imputed = triplets.to_csr();
        let Some(indicator) = self.indicator.as_ref() else {
            return Ok(imputed);
        };
        let mask = match indicator.transform(*features)? {
            SKIndicatorOutput::Sparse(mask) => mask,
            SKIndicatorOutput::Dense(mask) => {
                let mut triplets = TriMat::new(mask.dim());
                mask.indexed_iter()
                    .filter(|(_, value)| !value.is_zero())
                    .for_each(|((row, column), value)| triplets.add_triplet(row, column, *value));
                triplets.to_csr()
            }
        };
        Ok(horizontally_stack_sparse(&imputed, &mask))
    }
}
