//! TDD suite for large files and concurrent readers (task 4.5, PRD §8.7).

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::missing::SKMissingValuePolicy;
use crate::source::SKTabularSource;

const NUMBER_OF_ROWS: usize = 250_000;
const BATCH_SIZE: usize = 1_000;
const MEMORY_BUDGET_BYTES: usize = 1_048_576;
const NUMBER_OF_COLUMNS: usize = 3;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(
        (0..NUMBER_OF_COLUMNS)
            .map(|column| Field::new(format!("c{column}"), DataType::Float64, false))
            .collect::<Vec<_>>(),
    ))
}

fn write_large_parquet() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("large.parquet");
    let file = std::fs::File::create(&path).expect("a Parquet file");
    let mut writer = ArrowWriter::try_new(file, schema(), None).expect("a writer");
    for chunk_start in (0..NUMBER_OF_ROWS).step_by(10_000) {
        let rows = (chunk_start..(chunk_start + 10_000).min(NUMBER_OF_ROWS)).count();
        let columns: Vec<ArrayRef> = (0..NUMBER_OF_COLUMNS)
            .map(|column| {
                let values: Vec<f64> = (0..rows)
                    .map(|row| (chunk_start + row) as f64 + column as f64)
                    .collect();
                Arc::new(Float64Array::from(values)) as ArrayRef
            })
            .collect();
        let batch = RecordBatch::try_new(schema(), columns).expect("a batch");
        writer.write(&batch).expect("a written batch");
    }
    writer.close().expect("a finished file");
    (directory, path)
}

#[test]
fn sk_large_file_delivers_bounded_batches_and_every_row() {
    let (_directory, path) = write_large_parquet();
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Parquet)
        .with_batch_size(BATCH_SIZE)
        .with_missing_value_policy(SKMissingValuePolicy::Reject);

    source.open().expect("an open");
    let mut total_rows = 0;
    let mut batches = 0;
    while let Some(batch) = source.next_batch().expect("a batch") {
        let resident_bytes =
            batch.number_of_rows() * batch.number_of_columns() * std::mem::size_of::<f64>();
        assert!(
            resident_bytes < MEMORY_BUDGET_BYTES,
            "a batch of {resident_bytes} bytes exceeds the budget"
        );
        assert!(batch.number_of_rows() <= BATCH_SIZE);
        total_rows += batch.number_of_rows();
        batches += 1;
    }

    assert_eq!(total_rows, NUMBER_OF_ROWS);
    assert_eq!(batches, NUMBER_OF_ROWS / BATCH_SIZE);
}

#[test]
fn sk_concurrent_readers_produce_identical_results() {
    let (_directory, path) = write_large_parquet();
    let path = Arc::new(path);

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let path = path.clone();
            std::thread::spawn(move || {
                let mut source =
                    SKArrowFileSource::new(path.as_ref().clone(), SKArrowFileFormat::Parquet)
                        .with_batch_size(BATCH_SIZE);
                source.open().expect("an open");
                let mut rows = 0;
                let mut last = 0.0;
                while let Some(batch) = source.next_batch().expect("a batch") {
                    rows += batch.number_of_rows();
                    if let crate::columns::SKColumnData::Float64(column) =
                        batch.column(0).expect("the column")
                    {
                        last = column.last().copied().flatten().expect("a value");
                    }
                }
                (rows, last)
            })
        })
        .collect();

    for reader in readers {
        let (rows, last) = reader.join().expect("a joined reader");
        assert_eq!(rows, NUMBER_OF_ROWS);
        assert_eq!(last, (NUMBER_OF_ROWS - 1) as f64);
    }
}
