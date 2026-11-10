//! TDD suite for source projection pushdown and selection order (task 2.3).

use std::io::Write;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::source::SKTabularSource;

fn ten_column_parquet() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("wide.parquet");
    let fields: Vec<Field> = (0..10)
        .map(|column| Field::new(format!("c{column}"), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<ArrayRef> = (0..10)
        .map(|column| {
            let values: Vec<f64> = (0..20)
                .map(|row| (row as f64) * 100.0 + column as f64)
                .collect();
            Arc::new(Float64Array::from(values)) as ArrayRef
        })
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("a wide batch");

    let file = std::fs::File::create(&path).expect("a Parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("a Parquet writer");
    writer.write(&batch).expect("a written batch");
    writer.close().expect("a finished file");
    (directory, path)
}

#[test]
fn sk_parquet_projection_reads_only_the_selected_columns_in_selection_order() {
    let (_directory, path) = ten_column_parquet();
    let selection = SKColumnSelection::builder()
        .feature("c7")
        .feature("c2")
        .build()
        .expect("a selection");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Parquet)
        .with_selection(selection)
        .with_batch_size(7)
        .with_missing_value_policy(SKMissingValuePolicy::Reject);

    source.open().expect("an open");
    let batch = source
        .next_batch()
        .expect("a batch")
        .expect("a first batch");

    let names: Vec<&str> = batch
        .schema()
        .columns()
        .iter()
        .map(|column| column.name())
        .collect();
    assert_eq!(names, vec!["c7", "c2"]);
    assert_eq!(batch.number_of_columns(), 2);
    assert_eq!(batch.number_of_rows(), 7);

    let expected_c7: Vec<f64> = (0..7).map(|row| (row as f64) * 100.0 + 7.0).collect();
    let expected_c2: Vec<f64> = (0..7).map(|row| (row as f64) * 100.0 + 2.0).collect();
    assert_eq!(
        batch.column(0).expect("the first column"),
        &crate::columns::SKColumnData::Float64(expected_c7.into_iter().map(Some).collect())
    );
    assert_eq!(
        batch.column(1).expect("the second column"),
        &crate::columns::SKColumnData::Float64(expected_c2.into_iter().map(Some).collect())
    );
}

#[test]
fn sk_csv_projection_preserves_selection_order() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.csv");
    let mut file = std::fs::File::create(&path).expect("a CSV file");
    file.write_all(b"a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        .expect("CSV contents");
    let selection = SKColumnSelection::builder()
        .feature("c")
        .feature("a")
        .build()
        .expect("a selection");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv)
        .with_selection(selection)
        .with_batch_size(10)
        .with_missing_value_policy(SKMissingValuePolicy::Reject);

    source.open().expect("an open");
    let batch = source
        .next_batch()
        .expect("a batch")
        .expect("a first batch");
    let names: Vec<&str> = batch
        .schema()
        .columns()
        .iter()
        .map(|column| column.name())
        .collect();
    assert_eq!(names, vec!["c", "a"]);
    assert_eq!(
        batch.column(0).expect("the first column"),
        &crate::columns::SKColumnData::Float64(vec![Some(3.0), Some(6.0)])
    );
}

#[test]
fn sk_unknown_selection_column_is_rejected_before_reading() {
    let (_directory, path) = ten_column_parquet();
    let selection = SKColumnSelection::builder()
        .feature("missing")
        .build()
        .expect("a selection");
    let mut source =
        SKArrowFileSource::new(path, SKArrowFileFormat::Parquet).with_selection(selection);

    let error = source.open().expect_err("the column is absent");
    assert!(error.to_string().contains("missing"), "{error}");
}
