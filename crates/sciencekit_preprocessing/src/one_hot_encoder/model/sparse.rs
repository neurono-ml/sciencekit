//! The sparse (CSR) transform of the fitted one-hot encoder.

use rayon::prelude::*;
use sciencekit_common::{SKCategoricalView, SKError, SKFloat, sk_run_operation};
use sprs::CsMat;

use crate::one_hot_encoder::model::SKOneHotEncoderModel;

impl SKOneHotEncoderModel {
    /// Encode into a sparse (CSR) indicator matrix whose nonzero structure
    /// matches the dense transform exactly.
    ///
    /// Built in two passes: per-row output indices (parallel across rows), then
    /// the CSR `indptr`/`indices`/`data` arrays, validated by
    /// `CsMat::new_from_unsorted`.
    pub fn transform_sparse<'a, F, C>(&self, categories: C) -> Result<CsMat<F>, SKError>
    where
        F: SKFloat,
        C: TryInto<SKCategoricalView<'a>, Error = SKError>,
    {
        let view = categories.try_into()?;
        self.validate_shape(&view)?;
        sk_run_operation(self.operation_attributes(&view), || {
            let rows_count = view.number_of_rows();
            let row_indices: Vec<Vec<usize>> = (0..rows_count)
                .into_par_iter()
                .map(|row| self.resolve_row(&view, row))
                .collect::<Result<Vec<_>, SKError>>()?;

            let mut indptr = Vec::with_capacity(rows_count + 1);
            indptr.push(0);
            for row in &row_indices {
                let next = indptr.last().copied().unwrap_or(0) + row.len();
                indptr.push(next);
            }
            let total = indptr.last().copied().unwrap_or(0);
            let mut indices = Vec::with_capacity(total);
            let mut data = Vec::with_capacity(total);
            for row in &row_indices {
                for &column in row {
                    indices.push(column);
                    data.push(F::one());
                }
            }

            CsMat::new_from_unsorted(
                (rows_count, self.number_of_output_features),
                indptr,
                indices,
                data,
            )
            .map_err(|_| SKError::Conversion("CSR construction failed".into()))
        })
    }
}
