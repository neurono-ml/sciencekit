//! TDD suite for projection pushdown isolation, unselected unsupported columns
//! and self-describing format schema guards (tasks 2.3, 4.1, 4.3).

use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, TimestampMillisecondArray};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{SKArrowFileFormat, SKArrowFileSource};
use crate::columns::SKColumnData;
use crate::selection::SKColumnSelection;
use crate::source::SKTabularSource;

#[test]
fn sk_parquet_projection_ignores_unselected_temporal_and_list_columns() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("mixed.parquet");
    let mut list_builder = ListBuilder::new(Float64Builder::new());
    list_builder.values().append_value(1.0);
    list_builder.append(true);
    list_builder.values().append_slice(&[2.0, 3.0]);
    list_builder.append(true);
    let list_array: ArrayRef = Arc::new(list_builder.finish());

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("tags", list_array.data_type().clone(), true),
        Field::new("feature", DataType::Float64, false),
        Field::new("target", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(TimestampMillisecondArray::from(vec![1_000, 2_000])) as ArrayRef,
            list_array,
            Arc::new(Float64Array::from(vec![3.5, 7.5])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef,
        ],
    )
    .expect("a mixed Parquet batch");
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path).expect("a Parquet file"),
        schema,
        None,
    )
    .expect("a writer");
    writer.write(&batch).expect("a written batch");
    writer.close().expect("a closed file");

    let selection = SKColumnSelection::builder()
        .feature("feature")
        .target("target")
        .build()
        .expect("a selection");
    let mut source =
        SKArrowFileSource::new(path, SKArrowFileFormat::Parquet).with_selection(selection);
    source
        .open()
        .expect("unselected unsupported columns are ignored");
    let tabular = source
        .next_batch()
        .expect("a batch")
        .expect("a first batch");
    assert_eq!(
        tabular.column_named("feature"),
        Some(&SKColumnData::Float64(vec![Some(3.5), Some(7.5)]))
    );
    assert_eq!(
        tabular.column_named("target"),
        Some(&SKColumnData::Integer64(vec![Some(0), Some(1)]))
    );
}

#[test]
fn sk_ipc_projection_ignores_unselected_temporal_columns() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("mixed.arrow");
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("feature", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(TimestampMillisecondArray::from(vec![10, 20, 30])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.25, 2.5, 3.75])) as ArrayRef,
        ],
    )
    .expect("a mixed IPC batch");
    let mut writer = FileWriter::try_new(
        std::fs::File::create(&path).expect("an IPC file"),
        schema.as_ref(),
    )
    .expect("an IPC writer");
    writer.write(&batch).expect("a written batch");
    writer.finish().expect("a finished IPC file");

    let selection = SKColumnSelection::builder()
        .feature("feature")
        .build()
        .expect("a selection");
    let mut source =
        SKArrowFileSource::new(path, SKArrowFileFormat::IpcFile).with_selection(selection);
    source
        .open()
        .expect("unselected temporal column is ignored");
    let tabular = source
        .next_batch()
        .expect("a batch")
        .expect("a first batch");
    assert_eq!(
        tabular.column_named("feature"),
        Some(&SKColumnData::Float64(vec![
            Some(1.25),
            Some(2.5),
            Some(3.75)
        ]))
    );
}

#[test]
fn sk_parquet_unselected_column_chunk_is_genuinely_not_decoded() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("corrupted_column.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("unselected", DataType::Float64, false),
        Field::new("selected", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5])) as ArrayRef,
        ],
    )
    .expect("a two-column batch");
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path).expect("a Parquet file"),
        schema,
        None,
    )
    .expect("a Parquet writer");
    writer.write(&batch).expect("a written batch");
    writer.close().expect("a finished file");

    let probe =
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).expect("readable"))
            .expect("valid footer");
    let (start_offset, byte_length) = probe.metadata().row_group(0).column(0).byte_range();
    let mut bytes = std::fs::read(&path).expect("file bytes");
    bytes[start_offset as usize..(start_offset + byte_length) as usize].fill(0xFF);
    std::fs::write(&path, &bytes).expect("corrupted column chunk written");

    let mut unprojected = SKArrowFileSource::new(path.clone(), SKArrowFileFormat::Parquet);
    unprojected.open().expect("the footer remains valid");
    assert!(unprojected.next_batch().is_err());

    let selection = SKColumnSelection::builder()
        .feature("selected")
        .build()
        .expect("a selection");
    let mut projected =
        SKArrowFileSource::new(path, SKArrowFileFormat::Parquet).with_selection(selection);
    projected.open().expect("an open");
    let delivered = projected.next_batch().expect("decoded").expect("a batch");
    assert_eq!(
        delivered.column_named("selected"),
        Some(&SKColumnData::Float64(vec![
            Some(1.5),
            Some(2.5),
            Some(3.5)
        ]))
    );
}
