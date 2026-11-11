//! TDD suite for the JSON and NDJSON sources (task 4.4).

use arrow_schema::{DataType, Field, Schema};
use std::io::Write;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::missing::SKMissingValuePolicy;
use crate::source::SKTabularSource;

fn explicit_schema() -> Schema {
    Schema::new(vec![
        Field::new("value", DataType::Float64, true),
        Field::new("count", DataType::Int64, true),
    ])
}

fn write_file(name: &str, contents: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join(name);
    let mut file = std::fs::File::create(&path).expect("a JSON file");
    file.write_all(contents.as_bytes()).expect("JSON contents");
    (directory, path)
}

fn read_values(source: &mut SKArrowFileSource) -> Vec<f64> {
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
fn sk_json_with_explicit_schema_reads_the_document() {
    let (_directory, path) = write_file(
        "table.json",
        "{\"value\":1.5,\"count\":2}\n{\"value\":3.5,\"count\":4}\n",
    );
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Json)
        .with_explicit_schema(explicit_schema())
        .with_missing_value_policy(SKMissingValuePolicy::Reject);

    assert_eq!(read_values(&mut source), vec![1.5, 3.5]);
}

#[test]
fn sk_json_requires_a_schema_unless_inference_is_opted_in() {
    let (_directory, path) = write_file("table.json", "{\"value\":1.5}\n{\"value\":3.5}\n");
    let mut source = SKArrowFileSource::new(path.clone(), SKArrowFileFormat::Json);
    let error = source.open().expect_err("no schema and no inference");
    assert!(
        matches!(
            error,
            sciencekit_common::SKError::UnsupportedOperation { .. }
        ),
        "{error}"
    );

    let mut inferred =
        SKArrowFileSource::new(path, SKArrowFileFormat::Json).with_json_schema_inference(true);
    assert_eq!(read_values(&mut inferred), vec![1.5, 3.5]);
}

#[test]
fn sk_ndjson_with_explicit_schema_reads_one_record_per_line() {
    let (_directory, path) = write_file(
        "table.ndjson",
        "{\"value\":1.5,\"count\":2}\n{\"value\":3.5,\"count\":4}\n",
    );
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::NdJson)
        .with_explicit_schema(explicit_schema());

    assert_eq!(read_values(&mut source), vec![1.5, 3.5]);
}

#[test]
fn sk_malformed_json_document_is_a_structured_error() {
    let (_directory, path) = write_file("broken.json", "{\"value\":1.5,\"count\":2}\n{\"value\":");
    let mut source = SKArrowFileSource::new(path, SKArrowFileFormat::Json)
        .with_explicit_schema(explicit_schema());

    source.open().expect("the schema is explicit");
    let error = source.next_batch().expect_err("the document is truncated");
    assert!(
        matches!(error, sciencekit_common::SKError::Conversion(_)),
        "{error}"
    );
}
