//! The sparse (CSR) transform of the fitted polynomial-features model.

use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat, sk_run_operation};
use sprs::CsMat;

use crate::polynomial_features::model::SKPolynomialFeaturesModel;

impl SKPolynomialFeaturesModel {
    /// Expand into a sparse (CSR) matrix storing only the nonzero terms.
    ///
    /// A dense input is evaluated row by row; a sparse input is expanded
    /// directly from each CSR row, so neither path materializes an `n × p`
    /// matrix. The per-row nonzero entries (parallel across rows) feed a
    /// two-pass CSR build validated by `CsMat::new_from_unsorted`.
    pub fn transform_sparse<'a, F, X>(&self, features: X) -> Result<CsMat<F>, SKError>
    where
        F: SKFloat + Send + Sync,
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let (rows_count, columns_count) = match view {
            SKDataView::Dense(dense) => dense.dim(),
            SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
        self.validate_shape(columns_count, rows_count)?;
        sk_run_operation(self.operation_attributes(rows_count, columns_count), || {
            let row_entries: Vec<Vec<(usize, F)>> = match view {
                SKDataView::Dense(dense) => (0..rows_count)
                    .into_par_iter()
                    .map(|row_index| {
                        let row: Vec<F> = dense.row(row_index).to_vec();
                        self.evaluate_row(&row)
                    })
                    .collect(),
                SKDataView::Sparse(sparse) => (0..rows_count)
                    .into_par_iter()
                    .map(|row_index| {
                        let row = sparse.outer_view(row_index).expect("row within bounds");
                        self.evaluate_sparse_row(&row)
                    })
                    .collect(),
                _ => unreachable!("the shape check rejected every other representation"),
            };
            self.build_sparse(rows_count, row_entries)
        })
    }

    /// Evaluate every term for one CSR row without densifying it.
    fn evaluate_sparse_row<F: SKFloat>(&self, row: &sprs::CsVecView<'_, F>) -> Vec<(usize, F)> {
        self.terms
            .iter()
            .enumerate()
            .filter_map(|(term_index, term)| {
                let mut value = F::one();
                for &feature in term {
                    let position = row.indices().binary_search(&feature).ok()?;
                    value = value * row.data()[position];
                }
                (value != F::zero()).then_some((term_index, value))
            })
            .collect()
    }

    /// Assemble the per-row entries into one CSR matrix.
    fn build_sparse<F: SKFloat>(
        &self,
        rows_count: usize,
        row_entries: Vec<Vec<(usize, F)>>,
    ) -> Result<CsMat<F>, SKError> {
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
    }
}
