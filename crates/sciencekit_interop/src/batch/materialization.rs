//! Materialization of a heterogeneous table into homogeneous feature blocks
//! (spec `dataset-interop-core`).
//!
//! A column-major table cannot be aliased as a row-major matrix: the
//! conversion allocates a new block and copies each coerced column into its
//! place. Only an input already stored row-major is borrowed (see the
//! zero-copy views).

use ndarray::Array2;
use sciencekit_common::{SKError, SKFloat};

use super::{SKTabularBatch, sk_validate_numeric_feature_columns};
use crate::columns::sk_coerce_numeric_column;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

impl SKTabularBatch {
    /// Materialize the selected features as an owned row-major matrix.
    ///
    /// Feature order follows the selection order. The returned matrix owns a
    /// fresh buffer: the batch's column storage is never aliased.
    ///
    /// # Errors
    /// Propagates selection resolution, unsupported-type validation and
    /// missing-value policy failures.
    pub fn to_dense<F: SKFloat>(
        &self,
        selection: &SKColumnSelection,
        policy: SKMissingValuePolicy,
    ) -> Result<Array2<F>, SKError> {
        let resolved = selection.resolve(self.schema())?;
        sk_validate_numeric_feature_columns(self, resolved.feature_indices())?;

        let mut columns = Vec::with_capacity(resolved.number_of_features());
        for index in resolved.feature_indices() {
            let column_schema = &self.schema().columns()[*index];
            let values = sk_coerce_numeric_column::<F>(
                column_schema.name(),
                &self.columns()[*index],
                policy,
            )?;
            columns.push(values);
        }

        if self.number_of_rows() == 0 {
            return Ok(Array2::zeros((0, columns.len())));
        }

        let views: Vec<_> = columns.iter().map(|column| column.view()).collect();
        ndarray::stack(ndarray::Axis(1), &views)
            .map_err(|error| SKError::Conversion(format!("dense materialization failed: {error}")))
    }
}
