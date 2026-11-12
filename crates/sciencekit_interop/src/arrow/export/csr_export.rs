//! Export of CSR matrices as three Arrow arrays plus shape metadata (spec
//! `arrow-columnar-interop`).

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::{DataType, Field, Schema};
use sciencekit_common::{SKError, SKFloat};
use sprs::CsMatView;

use super::array_export::{sk_float_data_type, sk_primitive_array_from_slice};

/// Metadata keys carrying the logical shape of an exported CSR block.
pub const SK_ARROW_NUMBER_OF_ROWS_METADATA_KEY: &str = "sciencekit:number_of_rows";
/// Metadata key carrying the number of columns of an exported CSR block.
pub const SK_ARROW_NUMBER_OF_COLUMNS_METADATA_KEY: &str = "sciencekit:number_of_columns";

/// An exported compressed-row matrix: row pointers, column indices, stored
/// values and the logical shape.
///
/// Arrow has no sparse data type, so the three arrays are exported separately
/// (they have different lengths) with the shape attached to the schema.
#[derive(Debug, Clone)]
pub struct SKArrowCsrExport {
    schema: Schema,
    indptr: ArrayRef,
    indices: ArrayRef,
    data: ArrayRef,
    number_of_rows: usize,
    number_of_columns: usize,
    zero_copy: bool,
}

impl SKArrowCsrExport {
    /// The schema describing the three exported fields and the shape metadata.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The row-pointer array, length `rows + 1`.
    pub fn indptr(&self) -> &ArrayRef {
        &self.indptr
    }

    /// The column-index array, length `stored_values`.
    pub fn indices(&self) -> &ArrayRef {
        &self.indices
    }

    /// The stored-value array, length `stored_values`.
    pub fn data(&self) -> &ArrayRef {
        &self.data
    }

    /// The `(rows, columns)` logical shape.
    pub fn shape(&self) -> (usize, usize) {
        (self.number_of_rows, self.number_of_columns)
    }

    /// Whether the exported arrays alias the sparse source buffers.
    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy
    }
}

fn sk_i64_array(values: &[usize]) -> ArrayRef {
    let converted: Vec<i64> = values.iter().map(|value| *value as i64).collect();
    Arc::new(Int64Array::from(converted))
}

/// Export a CSR matrix as its three component arrays with shape metadata.
///
/// Row pointers, column indices and stored values are preserved verbatim, so
/// the block reconstructs exactly.
///
/// # Errors
/// Returns [`SKError::Conversion`] for a model scalar with no Arrow
/// counterpart.
pub fn sk_export_csr_matrix<F: SKFloat>(
    matrix: CsMatView<'_, F>,
) -> Result<SKArrowCsrExport, SKError> {
    let number_of_rows = matrix.rows();
    let number_of_columns = matrix.cols();
    let float_type = sk_float_data_type::<F>()?;
    let metadata = std::collections::HashMap::from([
        (
            SK_ARROW_NUMBER_OF_ROWS_METADATA_KEY.to_string(),
            number_of_rows.to_string(),
        ),
        (
            SK_ARROW_NUMBER_OF_COLUMNS_METADATA_KEY.to_string(),
            number_of_columns.to_string(),
        ),
    ]);
    let schema = Schema::new_with_metadata(
        vec![
            Field::new("indptr", DataType::Int64, false),
            Field::new("indices", DataType::Int64, false),
            Field::new("data", float_type, false),
        ],
        metadata,
    );

    Ok(SKArrowCsrExport {
        schema,
        indptr: sk_i64_array(matrix.indptr().raw_storage()),
        indices: sk_i64_array(matrix.indices()),
        data: sk_primitive_array_from_slice(matrix.data())?,
        number_of_rows,
        number_of_columns,
        zero_copy: false,
    })
}
