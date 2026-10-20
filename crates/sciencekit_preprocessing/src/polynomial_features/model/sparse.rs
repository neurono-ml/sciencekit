//! The sparse (CSR) transform of the fitted polynomial-features model.

use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat, sk_run_operation};
use sprs::CsMat;

use crate::polynomial_features::model::SKPolynomialFeaturesModel;

impl SKPolynomialFeaturesModel {
    /// Expand into a sparse (CSR) matrix storing only the nonzero terms.
    ///
    /// The per-row nonzero entries (parallel across rows) feed a two-pass CSR
    /// build validated by `CsMat::new_from_unsorted`.
    pub fn transform_sparse<'a, F, X>(&self, features: X) -> Result<CsMat<F>, SKError>
    where
        F: SKFloat + Send + Sync,
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        self.validate_shape(dense.ncols(), dense.nrows())?;
        sk_run_operation(
            self.operation_attributes(dense.nrows(), dense.ncols()),
            || {
                let rows_count = dense.nrows();
                let row_entries: Vec<Vec<(usize, F)>> = (0..rows_count)
                    .into_par_iter()
                    .map(|row_index| {
                        let row: Vec<F> = dense.row(row_index).to_vec();
                        self.evaluate_row(&row)
                    })
                    .collect();

                let mut indptr = Vec::with_capacity(rows_count + 1);
                indptr.push(0);
                for entries in &row_entries {
                    let next = indptr.last().copied().unwrap_or(0) + entries.len();
                    indptr.push(next);
                }
                let total = indptr.last().copied().unwrap_or(0);
                let mut indices = Vec::with_capacity(total);
                let mut data = Vec::with_capacity(total);
                for entries in &row_entries {
                    for &(term_index, value) in entries {
                        indices.push(term_index);
                        data.push(value);
                    }
                }

                CsMat::new_from_unsorted(
                    (rows_count, self.number_of_output_features()),
                    indptr,
                    indices,
                    data,
                )
                .map_err(|_| SKError::Conversion("CSR construction failed".into()))
            },
        )
    }
}
