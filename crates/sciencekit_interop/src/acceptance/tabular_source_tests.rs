//! Acceptance tests for the feature-gated tabular engines (spec
//! `tabular-sources`).

use sciencekit_common::SKFeatureBlock;

use crate::tabular::{SKColumnKind, SKTabularDataset, SKTabularSourceConfig};

fn sk_engine_config() -> SKTabularSourceConfig {
    SKTabularSourceConfig::builder()
        .column("x", SKColumnKind::Numeric)
        .column("label", SKColumnKind::Categorical)
        .missing_value_policy(crate::missing::SKMissingValuePolicy::NumericNaN)
        .build()
        .unwrap()
}

fn sk_dense(dataset: &SKTabularDataset<f64>) -> ndarray::Array2<f64> {
    match dataset.features() {
        SKFeatureBlock::Dense(matrix) => {
            matrix.mapv(|value| if value.is_nan() { -1.0 } else { value })
        }
        other => panic!("expected a dense block, found {other:?}"),
    }
}

#[cfg(feature = "polars")]
mod polars_acceptance {
    use ::polars::prelude::{IntoLazy, NamedFrom, Series, df};

    use super::*;
    use crate::polars::{SKPolarsLazySource, sk_tabular_dataset_from_dataframe};
    use crate::source::SKTabularSource;

    fn frame(rows: usize) -> ::polars::prelude::DataFrame {
        let x: Vec<Option<f64>> = (0..rows)
            .map(|row| if row % 97 == 0 { None } else { Some(row as f64) })
            .collect();
        let label: Vec<String> = (0..rows).map(|row| format!("label_{}", row % 4)).collect();
        df!(
            "x" => Series::new("x".into(), x),
            "label" => Series::new("label".into(), label),
            "ignored" => Series::new("ignored".into(), vec![0_i64; rows]),
        )
        .unwrap()
    }

    /// Tiny and large frames convert and read identically across threads.
    #[test]
    fn polars_acceptance_large_table_and_concurrent_reads() {
        for rows in [1_usize, 2_000] {
            let dataset =
                sk_tabular_dataset_from_dataframe::<f64>(&frame(rows), &sk_engine_config()).unwrap();
            assert_eq!(dataset.number_of_rows(), rows);
            assert_eq!(dataset.number_of_categorical_columns(), 1);
            let expected = sk_dense(&dataset);
            let shared = std::sync::Arc::new(dataset);
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let shared = std::sync::Arc::clone(&shared);
                    std::thread::spawn(move || sk_dense(&shared))
                })
                .collect();
            for handle in handles {
                assert_eq!(handle.join().unwrap(), expected);
            }
        }
    }

    /// Lazy batches keep order and row counts for a large frame.
    #[test]
    fn polars_acceptance_lazy_batches_keep_order() {
        let rows = 1_000;
        let mut source = SKPolarsLazySource::new(frame(rows).lazy(), sk_engine_config(), 128).unwrap();
        source.open().unwrap();
        let mut position = 0;
        let mut total_rows = 0;
        while let Some(batch) = source.next_batch().unwrap() {
            assert_eq!(batch.position(), position);
            total_rows += batch.number_of_rows();
            position += 1;
        }
        assert_eq!(total_rows, rows);
        assert_eq!(position, rows.div_ceil(128));
    }
}

#[cfg(feature = "datafusion")]
mod datafusion_acceptance {
    use std::sync::Arc;

    use ::datafusion::arrow::array::{ArrayRef, Float64Array, RecordBatch, StringArray};
    use ::datafusion::arrow::datatypes::{DataType, Field, Schema};
    use ::datafusion::prelude::SessionContext;

    use super::*;
    use crate::datafusion::{SKDataFusionStreamSource, sk_tabular_dataset_from_dataframe_blocking};
    use crate::source::SKTabularSource;

    fn record_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("label", DataType::Utf8, true),
            Field::new("ignored", DataType::Int64, true),
        ]));
        let x: Vec<Option<f64>> = (0..rows)
            .map(|row| if row % 97 == 0 { None } else { Some(row as f64) })
            .collect();
        let label: Vec<Option<String>> =
            (0..rows).map(|row| Some(format!("label_{}", row % 4))).collect();
        let ignored: Vec<Option<i64>> = (0..rows).map(|_| Some(0)).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(x)) as ArrayRef,
                Arc::new(StringArray::from(label)) as ArrayRef,
                Arc::new(::datafusion::arrow::array::Int64Array::from(ignored)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Tiny and large DataFrames convert and read identically across threads.
    #[test]
    fn datafusion_acceptance_large_table_and_concurrent_reads() {
        for rows in [1_usize, 2_000] {
            let context = SessionContext::new();
            let dataframe = context.read_batch(record_batch(rows)).unwrap();
            let dataset: SKTabularDataset<f64> =
                sk_tabular_dataset_from_dataframe_blocking(dataframe, &sk_engine_config()).unwrap();
            assert_eq!(dataset.number_of_rows(), rows);
            let expected = sk_dense(&dataset);
            let shared = std::sync::Arc::new(dataset);
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let shared = std::sync::Arc::clone(&shared);
                    std::thread::spawn(move || sk_dense(&shared))
                })
                .collect();
            for handle in handles {
                assert_eq!(handle.join().unwrap(), expected);
            }
        }
    }

    /// The streaming source yields every row exactly once in order.
    #[test]
    fn datafusion_acceptance_stream_preserves_rows() {
        let rows = 1_000;
        let context = SessionContext::new();
        let dataframe = context.read_batch(record_batch(rows)).unwrap();
        let mut source = SKDataFusionStreamSource::new(dataframe, sk_engine_config()).unwrap();
        source.open().unwrap();
        let mut total_rows = 0;
        let mut next_position = 0;
        while let Some(batch) = source.next_batch().unwrap() {
            assert_eq!(batch.position(), next_position);
            total_rows += batch.number_of_rows();
            next_position += 1;
        }
        assert_eq!(total_rows, rows);
    }
}
