//! TDD suite for the Parquet source (task 4.1).

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::missing::SKMissingValuePolicy;
use crate::source::SKTabularSource;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int64, false),
    ]))
}

fn write_parquet(rows: usize) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.parquet");
    let values: Vec<f64> = (0..rows).map(|row| row as f64).collect();
    let counts: Vec<i64> = (0..rows).map(|row| row as i64).collect();
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Float64Array::from(values)) as ArrayRef,
            Arc::new(Int64Array::from(counts)) as ArrayRef,
        ],
    )
    .expect("a batch");
    let file = std::fs::File::create(&path).expect("a Parquet file");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("a writer");
    writer.write(&batch).expect("a written batch");
    writer.close().expect("a finished file");
    (directory, path)
}

#[test]
fn sk_parquet_batches_never_exceed_the_configured_size() {
    let (_directory, path) = write_parquet(10);
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Parquet)
        .with_batch_size(4)
        .with_missing_value_policy(SKMissingValuePolicy::Reject);

    source.open().expect("an open");
    let mut sizes = Vec::new();
    while let Some(batch) = source.next_batch().expect("a batch") {
        sizes.push(batch.number_of_rows());
    }
    assert_eq!(sizes, vec![4, 4, 2]);
}

#[test]
fn sk_empty_parquet_file_yields_no_batch() {
    let (_directory, path) = write_parquet(0);
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Parquet).with_batch_size(8);

    source.open().expect("an open");
    assert!(source.next_batch().expect("the end").is_none());
}

#[test]
fn sk_truncated_parquet_file_is_a_structured_error() {
    let (directory, path) = write_parquet(50);
    let bytes = std::fs::read(&path).expect("the file bytes");
    let truncated = directory.path().join("truncated.parquet");
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).expect("a truncated file");

    let mut source = SKArrowFileSource::new(truncated, SKArrowFileFormat::Parquet);
    let error = source.open().expect_err("the footer is missing");
    assert!(
        matches!(error, sciencekit_common::SKError::Conversion(_)),
        "{error}"
    );
    assert!(error.to_string().contains("Parquet"), "{error}");
}

#[test]
fn sk_parquet_row_order_is_preserved_across_batches() {
    let (_directory, path) = write_parquet(9);
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Parquet).with_batch_size(3);

    source.open().expect("an open");
    let mut values = Vec::new();
    while let Some(batch) = source.next_batch().expect("a batch") {
        match batch.column(0).expect("the column") {
            crate::columns::SKColumnData::Float64(column) => {
                values.extend(column.iter().map(|value| value.expect("a value")));
            }
            other => panic!("unexpected column: {other:?}"),
        }
    }
    assert_eq!(values, (0..9).map(|row| row as f64).collect::<Vec<_>>());
}
