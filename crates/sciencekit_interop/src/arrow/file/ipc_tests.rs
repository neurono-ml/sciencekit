//! TDD suite for the Arrow IPC file and stream sources (task 4.3).

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow_ipc::writer::{FileWriter, StreamWriter};
use arrow_schema::{DataType, Field, Schema};

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::source::SKTabularSource;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int64, false),
    ]))
}

fn batches(number_of_batches: usize, rows_per_batch: usize) -> Vec<RecordBatch> {
    (0..number_of_batches)
        .map(|batch_index| {
            let values: Vec<f64> = (0..rows_per_batch)
                .map(|row| (batch_index * rows_per_batch + row) as f64)
                .collect();
            let counts: Vec<i64> = (0..rows_per_batch).map(|row| row as i64).collect();
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Float64Array::from(values)) as ArrayRef,
                    Arc::new(Int64Array::from(counts)) as ArrayRef,
                ],
            )
            .expect("a batch")
        })
        .collect()
}

fn write_ipc_file(batches: &[RecordBatch]) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrow");
    let file = std::fs::File::create(&path).expect("an IPC file");
    let mut writer = FileWriter::try_new(file, schema().as_ref()).expect("a file writer");
    for batch in batches {
        writer.write(batch).expect("a written batch");
    }
    writer.finish().expect("a finished file");
    (directory, path)
}

fn write_ipc_stream(batches: &[RecordBatch]) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrows");
    let file = std::fs::File::create(&path).expect("an IPC stream");
    let mut writer = StreamWriter::try_new(file, schema().as_ref()).expect("a stream writer");
    for batch in batches {
        writer.write(batch).expect("a written batch");
    }
    writer.finish().expect("a finished stream");
    (directory, path)
}

fn read_all(format: SKArrowFileFormat, path: std::path::PathBuf) -> Vec<f64> {
    let mut source = SKArrowFileSource::new(path, format).with_batch_size(2);
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
    values
}

#[test]
fn sk_ipc_file_round_trips_all_batches() {
    let (_directory, path) = write_ipc_file(&batches(2, 3));
    assert_eq!(
        read_all(SKArrowFileFormat::IpcFile, path),
        (0..6).map(|row| row as f64).collect::<Vec<_>>()
    );
}

#[test]
fn sk_ipc_stream_round_trips_all_batches() {
    let (_directory, path) = write_ipc_stream(&batches(2, 3));
    assert_eq!(
        read_all(SKArrowFileFormat::IpcStream, path),
        (0..6).map(|row| row as f64).collect::<Vec<_>>()
    );
}

#[test]
fn sk_truncated_ipc_stream_fails_after_the_complete_batches() {
    let (directory, path) = write_ipc_stream(&batches(3, 4));
    let bytes = std::fs::read(&path).expect("the stream bytes");
    let truncated = directory.path().join("truncated.arrows");
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).expect("a truncated stream");

    let mut source =
        SKArrowFileSource::new(truncated, SKArrowFileFormat::IpcStream).with_batch_size(4);
    source.open().expect("the schema is intact");

    let mut delivered = 0;
    let mut error = None;
    loop {
        match source.next_batch() {
            Ok(Some(_)) => delivered += 1,
            Ok(None) => break,
            Err(caught) => {
                error = Some(caught);
                break;
            }
        }
    }

    assert!(delivered >= 1, "complete batches are delivered first");
    let error = error.expect("the truncated tail fails");
    assert!(
        matches!(error, sciencekit_common::SKError::Conversion(_)),
        "{error}"
    );
}

#[test]
fn sk_ipc_projection_reads_only_the_selected_column() {
    let (_directory, path) = write_ipc_file(&batches(1, 5));
    let selection = crate::selection::SKColumnSelection::builder()
        .feature("count")
        .build()
        .expect("a selection");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::IpcFile)
        .with_selection(selection)
        .with_batch_size(5);

    source.open().expect("an open");
    let batch = source
        .next_batch()
        .expect("a batch")
        .expect("a first batch");
    assert_eq!(batch.number_of_columns(), 1);
    assert_eq!(batch.schema().columns()[0].name(), "count");
}
