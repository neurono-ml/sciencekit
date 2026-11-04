//! The in-memory sparse branch of the imputer's streaming fit: stack the
//! collected batches into one CSR and reuse the exact in-memory sparse `fit`.

use sciencekit_common::{SKDataBatch, SKDataView, SKError, SKFloat, SKUnsupervisedFit};
use sprs::CsMat;

use super::super::core_implementation::SKSimpleImputer;
use super::super::model::SKSimpleImputerModel;

impl<F: SKFloat> SKSimpleImputer<F> {
    /// Stack the collected batches into one CSR and run the exact sparse fit.
    pub(super) fn fit_collected_sparse(
        &self,
        batches: Vec<SKDataBatch<F>>,
    ) -> Result<SKSimpleImputerModel<F>, SKError> {
        let total_rows: usize = batches.iter().map(SKDataBatch::number_of_rows).sum();
        let columns_count = batches.first().map_or(0, SKDataBatch::number_of_columns);
        if batches.iter().all(|batch| batch.number_of_rows() == 0) {
            return Err(SKError::Conversion("streaming fit received no rows".into()));
        }
        let stacked = sk_stack_batches(&batches, total_rows, columns_count);
        self.fit(stacked.view())
    }
}

/// Vertically stack every batch (dense or sparse) into one CSR design.
fn sk_stack_batches<F: SKFloat>(
    batches: &[SKDataBatch<F>],
    total_rows: usize,
    columns_count: usize,
) -> CsMat<F> {
    let mut triplets = sprs::TriMat::new((total_rows, columns_count));
    let mut row_offset = 0;
    for batch in batches {
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
                    for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
                        triplets.add_triplet(row_offset + row, column, value);
                    }
                }
            }
            _ => unreachable!("the caller rejected every other representation"),
        }
        row_offset += batch.number_of_rows();
    }
    triplets.to_csr()
}
