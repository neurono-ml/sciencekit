//! The in-memory sparse branch of the robust scaler's streaming fit.
//!
//! The concatenated `fit` is dense-only, so an in-process stream with sparse
//! blocks stacks them into one CSR and reads the exact order statistics with the
//! implicit zeros as a virtual multiplicity. No dense `n × p` copy is built.

use sciencekit_common::{SKDataBatch, SKDataView, SKError, SKFloat};

use super::super::configuration::finalize_statistics;
use super::super::core_implementation::SKRobustScaler;
use super::super::model::SKRobustScalerModel;
use super::super::quantiles::sparse_column_quantile_statistics;
use super::super::sketch::{extend_sparse_sketches, new_sketches};

impl<F: SKFloat> SKRobustScaler<F> {
    /// Stack the collected batches into one CSR and finalize the exact model.
    pub(super) fn fit_collected_sparse(
        &self,
        batches: Vec<SKDataBatch<F>>,
    ) -> Result<SKRobustScalerModel<F>, SKError> {
        let total_rows: usize = batches.iter().map(SKDataBatch::number_of_rows).sum();
        let columns_count = batches.first().map_or(0, SKDataBatch::number_of_columns);
        let mut triplets = sprs::TriMat::new((total_rows, columns_count));
        let mut row_offset = 0;
        for batch in &batches {
            match batch.view() {
                SKDataView::Dense(data) => {
                    data.indexed_iter().for_each(|((row, column), &value)| {
                        if value != F::zero() {
                            triplets.add_triplet(row_offset + row, column, value);
                        }
                    });
                }
                SKDataView::Sparse(data) => {
                    for (row, sparse_row) in data.outer_iterator().enumerate() {
                        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data())
                        {
                            triplets.add_triplet(row_offset + row, column, value);
                        }
                    }
                }
                _ => unreachable!("the caller rejected every other representation"),
            }
            row_offset += batch.number_of_rows();
        }
        if total_rows == 0 {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        let stacked = triplets.to_csr();
        let (raw_center, raw_scale) = sparse_column_quantile_statistics(
            &stacked.view(),
            self.quantile_low,
            self.quantile_high,
        );
        let (center, scale) = finalize_statistics(&self.configuration(), &raw_center, &raw_scale);
        let mut sketches = new_sketches(columns_count);
        extend_sparse_sketches(&mut sketches, &stacked.view());
        Ok(SKRobustScalerModel::from_sketches(
            total_rows as u64,
            center,
            scale,
            sketches,
            self.configuration(),
            self.backend,
            self.execution_intent(),
        ))
    }
}
