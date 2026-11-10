//! TDD suite for exposing Arrow sources through the streaming providers
//! (task 2.4).

use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sciencekit_common::{
    SKLazySource, SKSourceProvider, SKSupervisedLazySource, SKSupervisedSourceProvider,
};

use super::{SKArrowOneShotTabularSource, SKArrowRecordBatchSource};
use crate::batch::SKFeatureLayout;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::source::SKTabularSourceProvider;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int64, false),
    ]))
}

fn reader_of(rows: usize) -> Box<dyn RecordBatchReader + Send> {
    let values: Vec<f64> = (0..rows).map(|row| row as f64).collect();
    let counts: Vec<i64> = (0..rows).map(|row| row as i64 * 10).collect();
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Float64Array::from(values)),
            Arc::new(arrow_array::Int64Array::from(counts)),
        ],
    )
    .expect("a well-formed batch");
    Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema(),
    ))
}

fn feature_selection() -> SKColumnSelection {
    SKColumnSelection::builder()
        .feature("value")
        .build()
        .expect("a feature selection")
}

fn supervised_selection() -> SKColumnSelection {
    SKColumnSelection::builder()
        .feature("value")
        .target("count")
        .build()
        .expect("a supervised selection")
}

#[test]
fn sk_reopenable_arrow_source_serves_multiple_epochs_through_the_provider() {
    let source = SKArrowRecordBatchSource::new(
        || Ok(reader_of(6)),
        Some(feature_selection()),
        2,
        SKMissingValuePolicy::Reject,
    )
    .expect("a source");
    let mut provider = SKTabularSourceProvider::<_, f64>::new(
        source,
        feature_selection(),
        SKMissingValuePolicy::Reject,
        SKFeatureLayout::Dense,
    );
    assert!(<SKTabularSourceProvider<SKArrowRecordBatchSource, f64> as SKSourceProvider<f64>>::IS_REOPENABLE);

    let first = collect_epoch(&mut provider);
    let second = collect_epoch(&mut provider);

    assert_eq!(first.len(), 3);
    assert_eq!(first, second);
    assert_eq!(first.last().expect("a batch").len(), 2);
}

fn collect_epoch(
    provider: &mut SKTabularSourceProvider<SKArrowRecordBatchSource, f64>,
) -> Vec<Vec<f64>> {
    let mut epoch = Vec::new();
    let mut pass = <SKTabularSourceProvider<SKArrowRecordBatchSource, f64> as SKSourceProvider<
        f64,
    >>::open(provider)
    .expect("an epoch");
    while let Some(batch) = pass.next_batch().expect("a batch") {
        epoch.push(
            batch
                .view()
                .as_dense()
                .expect("a dense view")
                .iter()
                .copied()
                .collect(),
        );
    }
    epoch
}

#[test]
fn sk_supervised_provider_reads_the_arrow_target_column() {
    let source = SKArrowRecordBatchSource::new(
        || Ok(reader_of(4)),
        Some(supervised_selection()),
        4,
        SKMissingValuePolicy::Reject,
    )
    .expect("a source");
    let mut provider = SKTabularSourceProvider::<_, f64>::new(
        source,
        supervised_selection(),
        SKMissingValuePolicy::Reject,
        SKFeatureLayout::Dense,
    );

    let mut pass =
        <SKTabularSourceProvider<SKArrowRecordBatchSource, f64> as SKSupervisedSourceProvider<
            f64,
        >>::open(&mut provider)
        .expect("an epoch");
    let batch = pass
        .next_supervised_batch()
        .expect("a batch")
        .expect("a supervised batch");
    assert_eq!(batch.number_of_rows(), 4);
    assert_eq!(batch.targets().to_vec(), vec![0.0, 10.0, 20.0, 30.0]);
}

#[test]
fn sk_one_shot_arrow_source_is_one_epoch_through_the_provider() {
    let source = SKArrowOneShotTabularSource::new(
        reader_of(3),
        Some(feature_selection()),
        2,
        SKMissingValuePolicy::Reject,
    )
    .expect("a source");
    let mut provider = SKTabularSourceProvider::<_, f64>::new(
        source,
        feature_selection(),
        SKMissingValuePolicy::Reject,
        SKFeatureLayout::Dense,
    );
    assert!(!<SKTabularSourceProvider<SKArrowOneShotTabularSource, f64> as SKSourceProvider<f64>>::IS_REOPENABLE);

    let mut pass =
        <SKTabularSourceProvider<SKArrowOneShotTabularSource, f64> as SKSourceProvider<f64>>::open(
            &mut provider,
        )
        .expect("an epoch");
    assert!(pass.next_batch().expect("a batch").is_some());

    let error = <SKTabularSourceProvider<SKArrowOneShotTabularSource, f64> as SKSourceProvider<
        f64,
    >>::open(&mut provider)
    .err()
    .expect("the second epoch fails");
    assert!(error.to_string().contains("one-shot"), "{error}");
}

#[test]
fn sk_arrow_provider_streams_sparse_blocks_without_densifying() {
    let source = SKArrowRecordBatchSource::new(
        || Ok(reader_of(3)),
        Some(feature_selection()),
        3,
        SKMissingValuePolicy::Reject,
    )
    .expect("a source");
    let mut provider = SKTabularSourceProvider::<_, f64>::new(
        source,
        feature_selection(),
        SKMissingValuePolicy::Reject,
        SKFeatureLayout::Sparse,
    );

    let mut pass = <SKTabularSourceProvider<SKArrowRecordBatchSource, f64> as SKSourceProvider<
        f64,
    >>::open(&mut provider)
    .expect("an epoch");
    let batch = pass.next_batch().expect("a batch").expect("a batch");
    assert!(batch.is_sparse());
    let view = batch.view();
    assert!(matches!(view, sciencekit_common::SKDataView::Sparse(_)));
}
