//! TDD suite for source projection order and positional selection (task 2.3).

use std::io::Write;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::source::SKTabularSource;

fn wide_batch(number_of_columns: usize, number_of_rows: usize) -> RecordBatch {
    let fields: Vec<Field> = (0..number_of_columns)
        .map(|column| Field::new(format!("c{column}"), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<ArrayRef> = (0..number_of_columns)
        .map(|column| {
            let values: Vec<f64> = (0..number_of_rows)
                .map(|row| (row as f64) * 100.0 + column as f64)
                .collect();
            Arc::new(Float64Array::from(values)) as ArrayRef
        })
        .collect();
    RecordBatch::try_new(schema, columns).expect("a wide batch")
}

fn write_parquet_batch(batch: &RecordBatch) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("wide.parquet");
    let file = std::fs::File::create(&path).expect("a Parquet file");
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).expect("a Parquet writer");
    writer.write(batch).expect("a written batch");
    writer.close().expect("a finished file");
    (directory, path)
}

fn write_ipc_batch(batch: &RecordBatch) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("wide.arrow");
    let file = std::fs::File::create(&path).expect("an IPC file");
    let mut writer =
        FileWriter::try_new(file, batch.schema().as_ref()).expect("an IPC file writer");
    writer.write(batch).expect("a written batch");
    writer.finish().expect("a finished IPC file");
    (directory, path)
}

#[test]
fn sk_parquet_projection_reads_only_the_selected_columns_in_selection_order() {
    let (_directory, path) = write_parquet_batch(&wide_batch(10, 20));
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

    let names: Vec<&str> = batch.schema().columns().iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["c7", "c2"]);
    assert_eq!(batch.number_of_rows(), 7);
    let expected_c7: Vec<Option<f64>> =
        (0..7).map(|row| Some((row as f64) * 100.0 + 7.0)).collect();
    let expected_c2: Vec<Option<f64>> =
        (0..7).map(|row| Some((row as f64) * 100.0 + 2.0)).collect();
    assert_eq!(batch.column(0), Some(&SKColumnData::Float64(expected_c7)));
    assert_eq!(batch.column(1), Some(&SKColumnData::Float64(expected_c2)));
}

#[test]
fn sk_positional_selection_resolves_once_for_parquet_and_ipc() {
    let wide = wide_batch(10, 6);
    let (_parquet_dir, parquet_path) = write_parquet_batch(&wide);
    let (_ipc_dir, ipc_path) = write_ipc_batch(&wide);
    let selection = SKColumnSelection::builder()
        .feature_at(7)
        .feature_at(2)
        .build()
        .expect("a positional selection");

    for (path, format) in [
        (parquet_path, SKArrowFileFormat::Parquet),
        (ipc_path, SKArrowFileFormat::IpcFile),
    ] {
        let mut source = SKArrowFileSource::new(path, format)
            .with_selection(selection.clone())
            .with_batch_size(6);
        source.open().expect("positional selection opens cleanly");
        let batch = source
            .next_batch()
            .expect("a batch")
            .expect("a first batch");
        let names: Vec<&str> = batch.schema().columns().iter().map(|c| c.name()).collect();
        assert_eq!(names, vec!["c7", "c2"]);
        let expected_c7: Vec<Option<f64>> =
            (0..6).map(|row| Some((row as f64) * 100.0 + 7.0)).collect();
        let expected_c2: Vec<Option<f64>> =
            (0..6).map(|row| Some((row as f64) * 100.0 + 2.0)).collect();
        assert_eq!(batch.column(0), Some(&SKColumnData::Float64(expected_c7)));
        assert_eq!(batch.column(1), Some(&SKColumnData::Float64(expected_c2)));
    }
}

#[test]
fn sk_reversed_positional_ipc_selection_preserves_selection_order() {
    let (_directory, path) = write_ipc_batch(&wide_batch(3, 4));
    let selection = SKColumnSelection::builder()
        .feature_at(1)
        .feature_at(0)
        .build()
        .expect("a reversed selection");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::IpcFile)
        .with_selection(selection)
        .with_batch_size(4);

    source.open().expect("an open");
    let batch = source
        .next_batch()
        .expect("a batch")
        .expect("a first batch");
    let names: Vec<&str> = batch.schema().columns().iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["c1", "c0"]);
    let expected_c1: Vec<Option<f64>> =
        (0..4).map(|row| Some((row as f64) * 100.0 + 1.0)).collect();
    let expected_c0: Vec<Option<f64>> = (0..4).map(|row| Some((row as f64) * 100.0)).collect();
    assert_eq!(batch.column(0), Some(&SKColumnData::Float64(expected_c1)));
    assert_eq!(batch.column(1), Some(&SKColumnData::Float64(expected_c0)));
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
    let names: Vec<&str> = batch.schema().columns().iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["c", "a"]);
    assert_eq!(
        batch.column(0),
        Some(&SKColumnData::Float64(vec![Some(3.0), Some(6.0)]))
    );
}

#[test]
fn sk_unknown_selection_column_is_rejected_before_reading() {
    let (_directory, path) = write_parquet_batch(&wide_batch(10, 20));
    let selection = SKColumnSelection::builder()
        .feature("missing")
        .build()
        .expect("a selection");
    let mut source =
        SKArrowFileSource::new(path, SKArrowFileFormat::Parquet).with_selection(selection);

    let error = source.open().expect_err("the column is absent");
    assert!(error.to_string().contains("missing"), "{error}");
}
