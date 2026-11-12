//! Tests for the Polars adapter (spec `tabular-sources`).

use ::polars::prelude::{DataType, IntoLazy, NamedFrom, Series, df};
use sciencekit_common::{SKError, SKFeatureBlock};

use super::{
    SKPolarsLazySource, sk_tabular_batch_from_dataframe, sk_tabular_dataset_from_dataframe,
};
use crate::missing::SKMissingValuePolicy;
use crate::source::SKTabularSource;
use crate::tabular::{SKColumnKind, SKTabularSourceConfig};

fn mixed_frame() -> ::polars::prelude::DataFrame {
    df!(
        "x" => &[1.0_f64, 2.0, 3.0],
        "count" => &[10_i32, 20, 30],
        "flag" => &[true, false, true],
        "label" => &["a", "b", "a"],
        "note" => &[9.0_f64, 9.0, 9.0],
    )
    .unwrap()
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

/// Numeric, integer and boolean columns form the feature block in config order.
#[test]
fn polars_dataframe_maps_numeric_columns_in_order() {
    let dataset =
        sk_tabular_dataset_from_dataframe::<f64>(&mixed_frame(), &mixed_config()).unwrap();
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => {
            assert_eq!(matrix[[0, 0]], 1.0);
            assert_eq!(matrix[[0, 1]], 10.0);
            assert_eq!(matrix[[0, 2]], 1.0);
            assert_eq!(matrix[[2, 2]], 1.0);
        }
        other => panic!("expected a dense block, found {other:?}"),
    }
    assert_eq!(dataset.number_of_rows(), 3);
}

/// Text columns form the owned categorical block with a label table.
#[test]
fn polars_dataframe_maps_categorical_columns() {
    let dataset =
        sk_tabular_dataset_from_dataframe::<f64>(&mixed_frame(), &mixed_config()).unwrap();
    let categorical = dataset.categorical().unwrap();
    assert_eq!(categorical.shape(), (3, 1));
    assert_eq!(categorical.codes().column(0).to_vec(), vec![0, 1, 0]);
    assert_eq!(categorical.label_table(0).unwrap().index_of("b"), Some(1));
}

/// Ignored columns are dropped from the batch and from the dataset schema.
#[test]
fn polars_dataframe_applies_column_selection() {
    let batch = sk_tabular_batch_from_dataframe(&mixed_frame(), &mixed_config(), 0, true).unwrap();
    assert!(batch.schema().column("note").is_none());
    assert_eq!(batch.number_of_columns(), 4);
    assert_eq!(batch.number_of_rows(), 3);
    assert!(batch.is_final());
}

/// Numeric nulls become NaN under the numeric policy.
#[test]
fn polars_dataframe_maps_numeric_nulls_to_nan() {
    let frame = df!("x" => &[Some(1.0_f64), None]).unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .missing_value_policy(SKMissingValuePolicy::NumericNaN)
        .build()
        .unwrap();
    let dataset = sk_tabular_dataset_from_dataframe::<f64>(&frame, &config).unwrap();
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => assert!(matrix[[1, 0]].is_nan()),
        other => panic!("expected a dense block, found {other:?}"),
    }
}

/// Categorical nulls become the documented missing sentinel.
#[test]
fn polars_dataframe_maps_categorical_nulls_to_sentinel() {
    let frame = df!("label" => &[Some("a"), None]).unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Categorical)
        .missing_value_policy(SKMissingValuePolicy::CategoricalMissingCode)
        .build()
        .unwrap();
    let dataset = sk_tabular_dataset_from_dataframe::<f64>(&frame, &config).unwrap();
    assert_eq!(dataset.categorical().unwrap().codes()[[1, 0]], -1);
}

/// A configured column absent from the frame is a structured error.
#[test]
fn polars_dataframe_rejects_an_unknown_column() {
    let config = SKTabularSourceConfig::builder()
        .column("absent", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error =
        sk_tabular_batch_from_dataframe(&mixed_frame(), &config, 0, true).unwrap_err();
    assert!(matches!(error, SKError::UnknownColumn { .. }));
}

/// A binary column has no neutral counterpart and is rejected by name.
#[test]
fn polars_dataframe_rejects_an_unsupported_dtype() {
    let frame = df!("blob" => &[b"a".as_slice(), b"b".as_slice()]).unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("blob", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let error =
        sk_tabular_batch_from_dataframe(&frame, &config, 0, true).unwrap_err();
    match error {
        SKError::UnsupportedColumnType { column, .. } => assert_eq!(column, "blob"),
        other => panic!("expected UnsupportedColumnType, found {other:?}"),
    }
}

/// A categorical dtype is read through its string labels.
#[test]
fn polars_dataframe_reads_categorical_dtype_as_text() {
    let series = Series::new("label".into(), &["a", "b", "a"])
        .cast(&DataType::Categorical(None, Default::default()))
        .unwrap();
    let frame = ::polars::prelude::DataFrame::new(vec![series.into()]).unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("label", SKColumnKind::Categorical)
        .build()
        .unwrap();
    let dataset = sk_tabular_dataset_from_dataframe::<f64>(&frame, &config).unwrap();
    assert_eq!(dataset.categorical().unwrap().codes()[[0, 0]], 0);
    assert_eq!(dataset.categorical().unwrap().codes()[[1, 0]], 1);
}

/// Lazy execution is sliced into ordered batches and can be reopened.
#[test]
fn polars_lazy_source_streams_ordered_batches() {
    let frame = df!(
        "x" => &[0.0_f64, 1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let config = SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .build()
        .unwrap();
    let mut source = SKPolarsLazySource::new(frame.lazy(), config, 2).unwrap();

    source.open().unwrap();
    let mut positions = Vec::new();
    let mut values = Vec::new();
    while let Some(batch) = source.next_batch().unwrap() {
        positions.push(batch.position());
        values.push(batch.columns()[0].clone());
    }
    assert_eq!(positions, vec![0, 1, 2]);
    assert_eq!(values.len(), 3);

    source.open().unwrap();
    let first_again = source.next_batch().unwrap().unwrap();
    assert_eq!(first_again.number_of_rows(), 2);
}
