//! TDD suite for the CSV source (task 4.2).

use arrow_schema::{DataType, Field, Schema};
use std::io::Write;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::source::SKTabularSource;

fn write_csv(contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.csv");
    let mut file = std::fs::File::create(&path).expect("a CSV file");
    file.write_all(contents.as_bytes()).expect("CSV contents");
    (directory, path)
}

fn first_batch(source: &mut SKArrowFileSource) -> crate::batch::SKTabularBatch {
    source.open().expect("an open");
    source
        .next_batch()
        .expect("a batch")
        .expect("a first batch")
}

#[test]
fn sk_csv_schema_is_inferred_from_the_header_by_default() {
    let (_directory, path) = write_csv("value,count\n1.5,10\n2.5,20\n");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv);

    let batch = first_batch(&mut source);
    let names: Vec<&str> = batch
        .schema()
        .columns()
        .iter()
        .map(|column| column.name())
        .collect();
    assert_eq!(names, vec!["value", "count"]);
    assert_eq!(batch.number_of_rows(), 2);
}

#[test]
fn sk_csv_explicit_schema_overrides_inference() {
    let (_directory, path) = write_csv("a,b\n1,2\n3,4\n");
    let schema = Schema::new(vec![
        Field::new("a", DataType::Float64, false),
        Field::new("b", DataType::Float64, false),
    ]);
    let mut source =
        SKArrowFileSource::new(path, SKArrowFileFormat::Csv).with_explicit_schema(schema);

    let batch = first_batch(&mut source);
    assert_eq!(
        batch.column(0).expect("the column"),
        &crate::columns::SKColumnData::Float64(vec![Some(1.0), Some(3.0)])
    );
}

#[test]
fn sk_csv_delimiter_and_header_options_are_honored() {
    let (_directory, path) = write_csv("1.0;2.0\n3.0;4.0\n");
    let schema = Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]);
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv)
        .with_explicit_schema(schema)
        .with_csv_header(false)
        .with_csv_delimiter(b';');

    let batch = first_batch(&mut source);
    assert_eq!(batch.number_of_rows(), 2);
    assert_eq!(
        batch.column(1).expect("the column"),
        &crate::columns::SKColumnData::Float64(vec![Some(2.0), Some(4.0)])
    );
}

#[test]
fn sk_csv_malformed_row_is_a_structured_error() {
    let (_directory, path) = write_csv("a,b\n1.0,2.0\nnot_a_number,4.0\n");
    let schema = Schema::new(vec![
        Field::new("a", DataType::Float64, false),
        Field::new("b", DataType::Float64, false),
    ]);
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv)
        .with_explicit_schema(schema)
        .with_batch_size(1);

    source.open().expect("an open");
    let first = source.next_batch().expect("a first batch");
    assert!(first.is_some(), "the first row is valid");
    let error = source.next_batch().expect_err("the malformed row surfaces");
    assert!(
        matches!(error, sciencekit_common::SKError::Conversion(_)),
        "{error}"
    );
}

#[test]
fn sk_csv_without_header_keeps_the_inferred_column_names() {
    let (_directory, path) = write_csv("1.0,2.0\n3.0,4.0\n");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Csv).with_csv_header(false);

    let batch = first_batch(&mut source);
    assert_eq!(batch.number_of_columns(), 2);
    assert_eq!(batch.number_of_rows(), 2);
    assert_eq!(
        batch.column(0).expect("the column"),
        &crate::columns::SKColumnData::Float64(vec![Some(1.0), Some(3.0)])
    );
}
