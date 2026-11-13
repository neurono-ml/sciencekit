//! Parquet prediction output through the Arrow/Parquet crates.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

use crate::error::SKCliError;

/// Write named double columns as one Parquet file.
pub fn sk_write_parquet_columns(
    path: &Path,
    names: &[String],
    columns: &[Vec<f64>],
) -> Result<(), SKCliError> {
    let fields: Vec<Field> = names
        .iter()
        .map(|name| Field::new(name, DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|column| Arc::new(Float64Array::from(column.clone())) as ArrayRef)
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|error| SKCliError::runtime(format!("Parquet batch assembly: {error}")))?;
    let file = File::create(path)
        .map_err(|error| SKCliError::data(path.display().to_string(), error.to_string()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|error| SKCliError::data(path.display().to_string(), error.to_string()))?;
    writer
        .write(&batch)
        .and_then(|()| writer.close())
        .map(|_| ())
        .map_err(|error| SKCliError::data(path.display().to_string(), error.to_string()))
}
