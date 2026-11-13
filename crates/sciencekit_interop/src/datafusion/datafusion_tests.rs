//! Tests for the DataFusion adapter (spec `tabular-sources`).

use ::datafusion::arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray,
};
use ::datafusion::arrow::datatypes::{DataType, Field, Schema};
use ::datafusion::prelude::SessionContext;
use sciencekit_common::{SKError, SKFeatureBlock};
use std::sync::Arc;

use super::{
    SKDataFusionStreamSource, sk_tabular_batch_from_datafusion_record_batch,
    sk_tabular_dataset_from_dataframe_blocking, sk_tabular_dataset_from_parquet_blocking,
};
use crate::missing::SKMissingValuePolicy;
use crate::source::SKTabularSource;
use crate::tabular::{SKColumnKind, SKTabularDataset, SKTabularSourceConfig};

fn mixed_record_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("count", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("label", DataType::Utf8, true),
        Field::new("note", DataType::Float64, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0)])),
        Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30)])),
        Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            Some(true),
        ])),
        Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("a")])),
        Arc::new(Float64Array::from(vec![Some(9.0), Some(9.0), Some(9.0)])),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}

fn mixed_config() -> SKTabularSourceConfig {
    SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Categorical)
        .column("x", SKColumnKind::Numeric)
        .column("count", SKColumnKind::Numeric)
        .column("flag", SKColumnKind::Boolean)
        .column("note", SKColumnKind::Ignored)
        .build()
        .unwrap()
}

/// Record batches map numeric, integer, boolean and text columns in order.
#[test]
fn datafusion_record_batch_maps_columns_in_config_order() {
    let batch = sk_tabular_batch_from_datafusion_record_batch(
        &mixed_record_batch(),
        &mixed_config(),
        0,
        true,
    )
    .unwrap();
    assert_eq!(batch.number_of_rows(), 3);
    assert_eq!(batch.number_of_columns(), 4);
    assert!(batch.schema().column("note").is_none());
    assert!(batch.is_final());
}

/// A DataFrame converts into a materialized dataset through the blocking path.
#[test]
fn datafusion_dataframe_converts_to_dataset() {
    let context = SessionContext::new();
    let dataframe = context.read_batch(mixed_record_batch()).unwrap();
    let dataset: SKTabularDataset<f64> =
        sk_tabular_dataset_from_dataframe_blocking(dataframe.clone(), &mixed_config()).unwrap();
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => {
            assert_eq!(matrix[[0, 0]], 1.0);
            assert_eq!(matrix[[1, 1]], 20.0);
            assert_eq!(matrix[[2, 2]], 1.0);
        }
        other => panic!("expected a dense block, found {other:?}"),
    }
    assert_eq!(dataset.categorical().unwrap().codes()[[1, 0]], 1);
}

/// The `TryFrom` surface accepts a borrowed DataFrame and config pair.
#[test]
fn datafusion_dataframe_try_from_pair() {
    let context = SessionContext::new();
    let dataframe = context.read_batch(mixed_record_batch()).unwrap();
    let dataset = SKTabularDataset::<f32>::try_from((&dataframe, &mixed_config())).unwrap();
    assert_eq!(dataset.number_of_rows(), 3);
}

/// Numeric and categorical nulls follow the configured policy.
#[test]
fn datafusion_missing_values_follow_the_policy() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("label", DataType::Utf8, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![Some(1.0), None])),
        Arc::new(StringArray::from(vec![Some("a"), None])),
    ];
    let record_batch = RecordBatch::try_new(schema, columns).unwrap();

    let numeric_config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .missing_value_policy(SKMissingValuePolicy::NumericNaN)
        .build()
        .unwrap();
    let numeric_batch =
        sk_tabular_batch_from_datafusion_record_batch(&record_batch, &numeric_config, 0, true)
            .unwrap();
    let numeric_dataset: SKTabularDataset<f64> =
        numeric_batch.to_tabular_dataset(&numeric_config).unwrap();
    match numeric_dataset.features() {
        SKFeatureBlock::Dense(matrix) => assert!(matrix[[1, 0]].is_nan()),
        other => panic!("expected a dense block, found {other:?}"),
    }

    let categorical_config = SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Categorical)
        .missing_value_policy(SKMissingValuePolicy::CategoricalMissingCode)
        .build()
        .unwrap();
    let categorical_batch =
        sk_tabular_batch_from_datafusion_record_batch(&record_batch, &categorical_config, 0, true)
            .unwrap();
    let categorical_dataset: SKTabularDataset<f64> = categorical_batch
        .to_tabular_dataset(&categorical_config)
        .unwrap();
    assert_eq!(
        categorical_dataset.categorical().unwrap().codes()[[1, 0]],
        -1
    );
}

/// A record batch materializes into the same schema and outputs as a DataFrame.
#[test]
fn datafusion_record_batch_matches_the_dataframe_outputs() {
    let config = mixed_config();
    let batch =
        sk_tabular_batch_from_datafusion_record_batch(&mixed_record_batch(), &config, 0, true)
            .unwrap();
    let from_batch: SKTabularDataset<f64> = batch.to_tabular_dataset(&config).unwrap();

    let context = SessionContext::new();
    let dataframe = context.read_batch(mixed_record_batch()).unwrap();
    let from_frame: SKTabularDataset<f64> =
        sk_tabular_dataset_from_dataframe_blocking(dataframe, &config).unwrap();

    assert_eq!(from_batch.schema(), from_frame.schema());
    assert_eq!(sk_dense_matrix(&from_batch), sk_dense_matrix(&from_frame));
    assert_eq!(
        from_batch.categorical().unwrap().codes(),
        from_frame.categorical().unwrap().codes()
    );
    assert_eq!(
        from_batch.categorical().unwrap().label_table(0),
        from_frame.categorical().unwrap().label_table(0)
    );
}

/// A configured column absent from the batch is a structured error.
#[test]
fn datafusion_record_batch_rejects_an_unknown_column() {
    let config = SKTabularSourceConfig::builder()
        .column("absent", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error =
        sk_tabular_batch_from_datafusion_record_batch(&mixed_record_batch(), &config, 0, true)
            .unwrap_err();
    assert!(matches!(error, SKError::UnknownColumn { .. }));
}

/// An unsupported dtype is rejected by column name.
#[test]
fn datafusion_record_batch_rejects_an_unsupported_dtype() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "when",
        DataType::Date32,
        true,
    )]));
    let record_batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Date32Array::from(vec![1])) as ArrayRef],
    )
    .unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("when", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error =
        sk_tabular_batch_from_datafusion_record_batch(&record_batch, &config, 0, true).unwrap_err();
    match error {
        SKError::UnsupportedColumnType { column, .. } => assert_eq!(column, "when"),
        other => panic!("expected UnsupportedColumnType, found {other:?}"),
    }
}

/// The streaming source preserves batch order and marks the final block.
#[test]
fn datafusion_stream_source_preserves_order_and_final_flag() {
    let context = SessionContext::new();
    let dataframe = context.read_batch(mixed_strided_batch()).unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let mut source = SKDataFusionStreamSource::new(dataframe, config).unwrap();

    source.open().unwrap();
    let mut positions = Vec::new();
    let mut finals = Vec::new();
    while let Some(batch) = source.next_batch().unwrap() {
        positions.push(batch.position());
        finals.push(batch.is_final());
    }
    assert!(!positions.is_empty());
    assert_eq!(positions.first(), Some(&0));
    assert_eq!(finals.last(), Some(&true));
    assert!(finals[..finals.len() - 1].iter().all(|is_final| !is_final));

    source.open().unwrap();
    assert!(source.next_batch().unwrap().is_some());
}

fn mixed_strided_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Float64Array::from(
            (0..6).map(|v| v as f64).collect::<Vec<_>>(),
        )) as ArrayRef],
    )
    .unwrap()
}

/// A Parquet file read through DataFusion uses the same conversion rules.
#[test]
fn datafusion_parquet_converts_with_the_same_rules() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.parquet");
    {
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ::datafusion::parquet::arrow::ArrowWriter::try_new(
            file,
            mixed_record_batch().schema(),
            None,
        )
        .unwrap();
        writer.write(&mixed_record_batch()).unwrap();
        writer.close().unwrap();
    }
    let dataset: SKTabularDataset<f64> =
        sk_tabular_dataset_from_parquet_blocking(&path, &mixed_config()).unwrap();
    assert_eq!(dataset.number_of_rows(), 3);
    assert_eq!(dataset.number_of_categorical_columns(), 1);
    assert_eq!(
        sk_dense_matrix(&dataset),
        ndarray::array![[1.0, 10.0, 1.0], [2.0, 20.0, 0.0], [3.0, 30.0, 1.0]]
    );
    assert_eq!(
        dataset.categorical().unwrap().codes().column(0).to_vec(),
        vec![0, 1, 0]
    );
    assert_eq!(dataset.schema().len(), 4);
}

/// A materialized dataset is shareable across threads and reads agree.
#[test]
fn datafusion_dataset_reads_agree_across_threads() {
    let context = SessionContext::new();
    let dataframe = context.read_batch(mixed_record_batch()).unwrap();
    let dataset: SKTabularDataset<f64> =
        sk_tabular_dataset_from_dataframe_blocking(dataframe, &mixed_config()).unwrap();
    let shared = Arc::new(dataset);
    let expected = sk_dense_matrix(&shared);

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || sk_dense_matrix(&shared))
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

fn sk_dense_matrix(dataset: &SKTabularDataset<f64>) -> ndarray::Array2<f64> {
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => matrix.clone(),
        other => panic!("expected a dense block, found {other:?}"),
    }
}

/// A streaming source refuses to nest its own runtime instead of panicking.
#[test]
fn datafusion_stream_source_refuses_to_nest_runtimes() {
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let constructor = runtime.block_on(async {
        let context = SessionContext::new();
        let dataframe = context.read_batch(mixed_strided_batch()).unwrap();
        SKDataFusionStreamSource::new(dataframe, config.clone()).map(|_| ())
    });
    match constructor {
        Err(SKError::UnsupportedOperation { operation, .. }) => {
            assert_eq!(operation, "datafusion_stream_source_new");
        }
        other => panic!("expected UnsupportedOperation from `new`, found {other:?}"),
    }

    let context = SessionContext::new();
    let dataframe = context.read_batch(mixed_strided_batch()).unwrap();
    let mut source = SKDataFusionStreamSource::new(dataframe, config).unwrap();
    let opened = runtime.block_on(async { source.open() });
    match opened {
        Err(SKError::UnsupportedOperation { operation, .. }) => {
            assert_eq!(operation, "datafusion_stream_open");
        }
        other => panic!("expected UnsupportedOperation from `open`, found {other:?}"),
    }
    let read = runtime.block_on(async { source.next_batch() });
    match read {
        Err(SKError::UnsupportedOperation { operation, .. }) => {
            assert_eq!(operation, "datafusion_stream_next_batch");
        }
        other => panic!("expected UnsupportedOperation from `next_batch`, found {other:?}"),
    }
}
