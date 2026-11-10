//! TDD suite for Arrow record-batch sources (task 2.1).

use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

use super::{SKArrowOneShotTabularSource, SKArrowRecordBatchSource};
use crate::missing::SKMissingValuePolicy;
use crate::source::SKTabularSource;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]))
}

fn batch_of(start: usize, rows: usize) -> RecordBatch {
    let values: Vec<f64> = (start..start + rows).map(|row| row as f64).collect();
    RecordBatch::try_new(schema(), vec![Arc::new(Float64Array::from(values))])
        .expect("a well-formed batch")
}

fn reader_of(batches: Vec<RecordBatch>) -> Box<dyn RecordBatchReader + Send> {
    Box::new(RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        schema(),
    ))
}

#[test]
fn sk_batch_size_is_respected_and_only_the_last_batch_is_final() {
    let source = SKArrowRecordBatchSource::new(
        || Ok(reader_of(vec![batch_of(0, 10_000)])),
        None,
        1_000,
        SKMissingValuePolicy::Reject,
    )
    .expect("a valid source");
    let mut source = source;
    source.open().expect("a first open");

    let mut delivered = Vec::new();
    while let Some(batch) = source.next_batch().expect("a batch") {
        delivered.push(batch);
    }

    assert_eq!(delivered.len(), 10);
    for (position, batch) in delivered.iter().enumerate() {
        assert!(batch.number_of_rows() <= 1_000);
        assert_eq!(batch.number_of_rows(), 1_000);
        assert_eq!(batch.position(), position);
        assert_eq!(batch.is_final(), position == delivered.len() - 1);
    }
}

#[test]
fn sk_partial_last_batch_is_marked_final() {
    let source = SKArrowRecordBatchSource::new(
        || Ok(reader_of(vec![batch_of(0, 25)])),
        None,
        10,
        SKMissingValuePolicy::Reject,
    )
    .expect("a valid source");
    let mut source = source;
    source.open().expect("a first open");

    let mut sizes = Vec::new();
    while let Some(batch) = source.next_batch().expect("a batch") {
        sizes.push((batch.number_of_rows(), batch.is_final()));
    }
    assert_eq!(sizes, vec![(10, false), (10, false), (5, true)]);
}

#[test]
fn sk_reader_error_propagates_after_complete_batches() {
    struct FailingReader {
        schema: SchemaRef,
        emitted: bool,
    }

    impl Iterator for FailingReader {
        type Item = Result<RecordBatch, ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.emitted {
                return Some(Err(ArrowError::ComputeError("reader failed".to_string())));
            }
            self.emitted = true;
            Some(Ok(batch_of(0, 2)))
        }
    }

    impl RecordBatchReader for FailingReader {
        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    let reader: Box<dyn RecordBatchReader + Send> = Box::new(FailingReader {
        schema: schema(),
        emitted: false,
    });
    let mut source =
        SKArrowOneShotTabularSource::new(reader, None, 2, SKMissingValuePolicy::Reject)
            .expect("a valid source");
    source.open().expect("a first open");

    let first = source
        .next_batch()
        .expect("a first batch")
        .expect("a batch");
    assert_eq!(first.number_of_rows(), 2);
    let error = source.next_batch().expect_err("the reader error surfaces");
    assert!(error.to_string().contains("reader failed"), "{error}");
}

#[test]
fn sk_zero_batch_size_is_rejected() {
    let error = SKArrowRecordBatchSource::new(
        || Ok(reader_of(Vec::new())),
        None,
        0,
        SKMissingValuePolicy::Reject,
    )
    .err()
    .expect("a zero batch size is invalid");
    assert!(error.to_string().contains("batch_size"), "{error}");
}

#[test]
fn sk_one_shot_source_rejects_a_second_open() {
    let mut source = SKArrowOneShotTabularSource::new(
        reader_of(vec![batch_of(0, 3)]),
        None,
        3,
        SKMissingValuePolicy::Reject,
    )
    .expect("a valid source");
    assert!(!SKArrowOneShotTabularSource::IS_REOPENABLE);

    source.open().expect("a first open");
    assert!(source.next_batch().expect("a batch").is_some());
    assert!(source.next_batch().expect("the end").is_none());

    let error = source.open().expect_err("the second open fails");
    assert!(error.to_string().contains("already consumed"), "{error}");
}

#[test]
fn sk_reopenable_source_replays_the_same_batches() {
    let source = SKArrowRecordBatchSource::new(
        || Ok(reader_of(vec![batch_of(0, 5), batch_of(5, 5)])),
        None,
        4,
        SKMissingValuePolicy::Reject,
    )
    .expect("a valid source");
    let mut source = source;

    let mut passes = Vec::new();
    for _ in 0..2 {
        source.open().expect("an open");
        let mut pass = Vec::new();
        while let Some(batch) = source.next_batch().expect("a batch") {
            pass.push(batch.number_of_rows());
        }
        passes.push(pass);
    }

    assert_eq!(passes[0], vec![4, 1, 4, 1]);
    assert_eq!(passes[0], passes[1]);
}
