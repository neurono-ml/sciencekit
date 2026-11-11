//! Export of dense matrices as Arrow arrays (spec `arrow-columnar-interop`).

use std::sync::Arc;

use arrow_array::{ArrayRef, StructArray};
use arrow_schema::{Field, Fields};
use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};

use super::array_export::{sk_float_data_type, sk_primitive_array_from_slice};

fn sk_column_values<F: SKFloat>(matrix: ArrayView2<F>, column: usize) -> Vec<F> {
    matrix.column(column).iter().copied().collect()
}

/// Export a dense matrix view of any memory stride as an Arrow array.
///
/// A one-column matrix becomes a primitive `f32`/`f64` array. A zero-column or
/// multi-column matrix becomes a `StructArray` with one field per column and
/// length equal to `matrix.nrows()`, because Arrow stores columns contiguously.
/// Values are copied column-by-column into Arrow buffers, so row-major,
/// column-major and sliced views are all accepted.
///
/// # Errors
/// Returns [`SKError::Conversion`] when the model scalar has no Arrow
/// counterpart or Arrow rejects the constructed `StructArray`.
pub fn sk_export_dense_matrix<F: SKFloat>(
    matrix: ArrayView2<F>,
    name: &str,
) -> Result<ArrayRef, SKError> {
    let float_type = sk_float_data_type::<F>()?;
    let number_of_columns = matrix.ncols();
    if number_of_columns == 1 {
        let values = sk_column_values(matrix, 0);
        return sk_primitive_array_from_slice(&values);
    }

    let mut fields = Vec::with_capacity(number_of_columns);
    let mut columns = Vec::with_capacity(number_of_columns);
    for column in 0..number_of_columns {
        let values = sk_column_values(matrix, column);
        fields.push(Field::new(
            format!("{name}_{column}"),
            float_type.clone(),
            true,
        ));
        columns.push(sk_primitive_array_from_slice(&values)?);
    }
    let structure =
        StructArray::try_new_with_length(Fields::from(fields), columns, None, matrix.nrows())
            .map_err(|error| {
                SKError::Conversion(format!("Arrow dense-matrix export failed: {error}"))
            })?;
    Ok(Arc::new(structure))
}
