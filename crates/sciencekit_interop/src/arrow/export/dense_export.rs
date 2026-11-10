//! Export of dense matrices as Arrow arrays (spec `arrow-columnar-interop`).

use std::any::TypeId;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Float64Array, StructArray};
use arrow_schema::{DataType, Field, Fields};
use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};

fn sk_export_error() -> SKError {
    SKError::Conversion(
        "the model scalar is neither f32 nor f64, so it has no Arrow counterpart".to_string(),
    )
}

fn sk_column_values<F: SKFloat>(matrix: ArrayView2<F>, column: usize) -> Vec<F> {
    matrix.column(column).iter().copied().collect()
}

fn sk_f32_array(values: &[f32]) -> ArrayRef {
    Arc::new(Float32Array::from(values.to_vec()))
}

fn sk_f64_array(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}

/// Export a dense matrix as an Arrow array.
///
/// A one-column matrix becomes a primitive `f32`/`f64` array. A wider matrix
/// becomes a `StructArray` with one field per column, because Arrow stores
/// columns contiguously and cannot alias a row-major matrix. The values are
/// copied into Arrow buffers; export from an owned block that can move its
/// buffer is a separate path.
///
/// # Errors
/// Returns [`SKError::Conversion`] for a model scalar with no Arrow
/// counterpart and [`SKError::UnsupportedRepresentation`] for a column-major
/// matrix.
pub fn sk_export_dense_matrix<F: SKFloat>(
    matrix: ArrayView2<F>,
    name: &str,
) -> Result<ArrayRef, SKError> {
    if !matrix.is_standard_layout() {
        return Err(SKError::UnsupportedRepresentation {
            representation: "column_major",
            suggestion: "reorder the matrix to row-major before exporting it to Arrow",
        });
    }

    let number_of_columns = matrix.ncols();
    if number_of_columns == 1 {
        let values = sk_column_values(matrix, 0);
        return if TypeId::of::<F>() == TypeId::of::<f32>() {
            let converted: Vec<f32> = values.iter().map(|value| value.to_f32().unwrap()).collect();
            Ok(sk_f32_array(&converted))
        } else if TypeId::of::<F>() == TypeId::of::<f64>() {
            let converted: Vec<f64> = values.iter().map(|value| value.to_f64().unwrap()).collect();
            Ok(sk_f64_array(&converted))
        } else {
            Err(sk_export_error())
        };
    }

    let mut fields = Vec::with_capacity(number_of_columns);
    let mut columns = Vec::with_capacity(number_of_columns);
    for column in 0..number_of_columns {
        let values = sk_column_values(matrix, column);
        let field_name = format!("{name}_{column}");
        if TypeId::of::<F>() == TypeId::of::<f32>() {
            let converted: Vec<f32> = values.iter().map(|value| value.to_f32().unwrap()).collect();
            fields.push(Field::new(field_name, DataType::Float32, true));
            columns.push(sk_f32_array(&converted));
        } else if TypeId::of::<F>() == TypeId::of::<f64>() {
            let converted: Vec<f64> = values.iter().map(|value| value.to_f64().unwrap()).collect();
            fields.push(Field::new(field_name, DataType::Float64, true));
            columns.push(sk_f64_array(&converted));
        } else {
            return Err(sk_export_error());
        }
    }
    Ok(Arc::new(StructArray::new(
        Fields::from(fields),
        columns,
        None,
    )))
}
