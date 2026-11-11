//! Export of owned streaming blocks as Arrow record batches (spec
//! `arrow-columnar-interop`).

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema};
use ndarray::ArrayView2;
use sciencekit_common::{SKDataBatch, SKDataView, SKError, SKFloat, SKSupervisedBatch};

use super::array_export::{sk_float_data_type, sk_primitive_array_from_slice};
use super::csr_export::{SKArrowCsrExport, sk_export_csr_matrix};

/// An exported record batch plus the conversion report.
#[derive(Debug, Clone)]
pub struct SKArrowExportedRecordBatch {
    batch: RecordBatch,
    zero_copy: bool,
}

impl SKArrowExportedRecordBatch {
    /// The exported record batch.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Whether the batch aliases the source buffers (always false for a copy).
    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy
    }
}

/// The export of a streaming block, whichever representation it carried.
#[derive(Debug, Clone)]
pub enum SKArrowBatchExport {
    /// A dense block exported as feature columns.
    Dense(SKArrowExportedRecordBatch),
    /// A sparse block exported as row pointers, column indices, values and a
    /// shape.
    Sparse(SKArrowCsrExport),
}

fn sk_dense_record_batch<F: SKFloat>(
    matrix: ArrayView2<F>,
    target: Option<(&str, ndarray::ArrayView1<F>)>,
) -> Result<RecordBatch, SKError> {
    let float_type = sk_float_data_type::<F>()?;
    let mut fields = Vec::with_capacity(matrix.ncols() + usize::from(target.is_some()));
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.capacity());
    for column in 0..matrix.ncols() {
        let values: Vec<F> = matrix.column(column).iter().copied().collect();
        fields.push(Field::new(
            format!("feature_{column}"),
            float_type.clone(),
            true,
        ));
        columns.push(sk_primitive_array_from_slice(&values)?);
    }
    if let Some((name, targets)) = target {
        let values: Vec<F> = targets.iter().copied().collect();
        fields.push(Field::new(name, float_type, true));
        columns.push(sk_primitive_array_from_slice(&values)?);
    }
    let options = RecordBatchOptions::new().with_row_count(Some(matrix.nrows()));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
        .map_err(|error| SKError::Conversion(format!("Arrow record-batch export failed: {error}")))
}

/// Export a streaming data block as either a dense record batch or the three
/// arrays of a sparse block.
///
/// # Errors
/// Propagates the dense or sparse export failures.
pub fn sk_export_data_batch<F: SKFloat>(
    batch: &SKDataBatch<F>,
) -> Result<SKArrowBatchExport, SKError> {
    match batch.view() {
        SKDataView::Dense(matrix) => Ok(SKArrowBatchExport::Dense(SKArrowExportedRecordBatch {
            batch: sk_dense_record_batch(matrix, None)?,
            zero_copy: false,
        })),
        SKDataView::Sparse(matrix) => Ok(SKArrowBatchExport::Sparse(sk_export_csr_matrix(matrix)?)),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "export the block as dense or CSR before converting it to Arrow",
        }),
    }
}

/// Export a supervised streaming block as feature columns plus a target
/// column.
///
/// # Errors
/// Returns [`SKError::UnsupportedRepresentation`] for a sparse supervised
/// block, which has no single rectangular Arrow shape, and propagates the dense
/// export failures. Export the feature block with [`sk_export_data_batch`] and
/// the targets as a prediction array instead.
pub fn sk_export_supervised_batch<F: SKFloat>(
    batch: &SKSupervisedBatch<F>,
    target_name: &str,
) -> Result<SKArrowExportedRecordBatch, SKError> {
    match batch.view() {
        SKDataView::Dense(matrix) => Ok(SKArrowExportedRecordBatch {
            batch: sk_dense_record_batch(matrix, Some((target_name, batch.targets())))?,
            zero_copy: false,
        }),
        SKDataView::Sparse(_) => Err(SKError::UnsupportedRepresentation {
            representation: "sparse_supervised",
            suggestion: "export the feature block with sk_export_data_batch and the targets as a prediction array",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "export the feature block with sk_export_data_batch and the targets as a prediction array",
        }),
    }
}
