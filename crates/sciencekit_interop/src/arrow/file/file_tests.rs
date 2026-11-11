//! TDD suite for reopenable file sources (task 2.2).

use std::io::Write;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::missing::SKMissingValuePolicy;
use crate::source::SKTabularSource;

fn write_csv(contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.csv");
    let mut file = std::fs::File::create(&path).expect("a CSV file");
    file.write_all(contents.as_bytes()).expect("CSV contents");
    (directory, path)
}

fn collect_pass(source: &mut SKArrowFileSource) -> Vec<crate::batch::SKTabularBatch> {
    source.open().expect("an open");
    let mut batches = Vec::new();
    while let Some(batch) = source.next_batch().expect("a batch") {
        batches.push(batch);
    }
    batches
}

#[test]
fn sk_file_source_replays_identical_batches_on_every_pass() {
    let (_directory, path) = write_csv("value,count\n1,10\n2,20\n3,30\n4,40\n5,50\n");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv)
        .with_batch_size(2)
        .with_missing_value_policy(SKMissingValuePolicy::Reject);

    let first = collect_pass(&mut source);
    let second = collect_pass(&mut source);

    assert_eq!(first.len(), 3);
    assert_eq!(first, second);
    assert_eq!(
        first
            .iter()
            .map(|batch| batch.number_of_rows())
            .collect::<Vec<_>>(),
        vec![2, 2, 1]
    );
    assert!(first.last().expect("a batch").is_final());
    assert!(first[..2].iter().all(|batch| !batch.is_final()));
}

#[test]
fn sk_file_source_keeps_row_order_across_passes() {
    let (_directory, path) = write_csv("value\n1.0\n2.0\n3.0\n4.0\n");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv).with_batch_size(1);

    let batches = collect_pass(&mut source);
    let values: Vec<f64> = batches
        .iter()
        .map(|batch| match batch.column(0).expect("the column") {
            crate::columns::SKColumnData::Float64(values) => values[0].expect("a value"),
            other => panic!("unexpected column: {other:?}"),
        })
        .collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn sk_missing_file_is_an_io_error() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("absent.parquet");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Parquet);

    let error = source.open().expect_err("the file does not exist");
    assert!(
        matches!(error, sciencekit_common::SKError::Io(_)),
        "{error}"
    );
}

#[test]
fn sk_zero_batch_size_is_rejected_at_open() {
    let (_directory, path) = write_csv("value\n1\n");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv).with_batch_size(0);

    let error = source.open().expect_err("a zero batch size is invalid");
    assert!(error.to_string().contains("batch_size"), "{error}");
}

#[test]
fn sk_self_describing_formats_reject_an_explicit_schema() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("dummy.parquet");
    let explicit = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "value",
        arrow_schema::DataType::Float64,
        false,
    )]);
    for format in [
        SKArrowFileFormat::Parquet,
        SKArrowFileFormat::IpcFile,
        SKArrowFileFormat::IpcStream,
    ] {
        let mut source =
            SKArrowFileSource::new(path.clone(), format).with_explicit_schema(explicit.clone());
        let error = source.open().expect_err("explicit schema rejected");
        assert!(
            matches!(
                error,
                sciencekit_common::SKError::UnsupportedOperation { .. }
            ),
            "{error}"
        );
    }
}
