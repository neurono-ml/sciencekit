//! Companion tests for the OpenTelemetry metric bridge (spec `telemetry-export`).

use std::collections::HashMap;

use ndarray::array;
use opentelemetry::Value;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExecutionMode, SKFeatureTransformer, SKUnsupervisedFit,
};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use tracing_subscriber::prelude::*;

use super::{
    SK_METRIC_OPERATION_COLUMNS, SK_METRIC_OPERATION_COUNT, SK_METRIC_OPERATION_DURATION,
    SK_METRIC_OPERATION_ERRORS, SK_METRIC_OPERATION_ROWS, sk_build_meter_provider,
    sk_build_metrics_layer,
};
use crate::configuration::SKTelemetryConfiguration;

fn attribute_map(attributes: &[opentelemetry::KeyValue]) -> HashMap<String, String> {
    attributes
        .iter()
        .map(|entry| {
            let text = match &entry.value {
                Value::String(value) => value.to_string(),
                other => format!("{other:?}"),
            };
            (entry.key.as_str().to_string(), text)
        })
        .collect()
}

fn find_metric<'a>(
    exported: &'a [ResourceMetrics],
    metric_name: &str,
) -> &'a opentelemetry_sdk::metrics::data::Metric {
    exported
        .iter()
        .flat_map(|resource_metrics| resource_metrics.scope_metrics())
        .flat_map(|scope_metrics| scope_metrics.metrics())
        .find(|metric| metric.name() == metric_name)
        .unwrap_or_else(|| panic!("expected metric `{metric_name}` to be exported"))
}

#[test]
fn completed_operations_export_duration_count_rows_and_columns_metrics_with_labels() {
    let exporter = InMemoryMetricExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-metrics-test")
        .build()
        .expect("configuration should build");
    let provider = sk_build_meter_provider(&configuration, exporter.clone());
    let subscriber = tracing_subscriber::registry().with(sk_build_metrics_layer(&provider));

    let train = array![
        [1.0_f64, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];
    tracing::subscriber::with_default(subscriber, || {
        let scaler = SKStandardScaler::build()
            .backend(SKBackendKind::Faer)
            .execution_mode(SKExecutionMode::InProcessSynchronous)
            .build()
            .expect("scaler should build");
        let fitted = scaler.fit(&train).expect("fit should succeed");
        let _transformed = fitted.transform(&train).expect("transform should succeed");
    });

    provider.force_flush().expect("metric flush should succeed");
    let finished = exporter.get_finished_metrics().expect("metrics readable");

    let duration_metric = find_metric(&finished, SK_METRIC_OPERATION_DURATION);
    assert_eq!(duration_metric.unit(), "ms");
    let AggregatedMetrics::F64(MetricData::Histogram(duration_histogram)) = duration_metric.data()
    else {
        panic!("expected f64 histogram for duration");
    };
    let duration_points: Vec<_> = duration_histogram.data_points().collect();
    assert_eq!(duration_points.len(), 2);
    let fit_point = duration_points
        .iter()
        .find(|point| {
            let attrs: Vec<_> = point.attributes().cloned().collect();
            attribute_map(&attrs).get("operation").map(String::as_str) == Some("fit")
        })
        .expect("fit duration point should exist");
    let fit_attrs: Vec<_> = fit_point.attributes().cloned().collect();
    let fit_labels = attribute_map(&fit_attrs);
    assert_eq!(fit_labels.get("backend").map(String::as_str), Some("faer"));
    assert_eq!(
        fit_labels.get("mode").map(String::as_str),
        Some("InProcessSynchronous")
    );

    let count_metric = find_metric(&finished, SK_METRIC_OPERATION_COUNT);
    let AggregatedMetrics::U64(MetricData::Sum(count_sum)) = count_metric.data() else {
        panic!("expected u64 counter for operation count");
    };
    let total_operations: u64 = count_sum.data_points().map(|point| point.value()).sum();
    assert_eq!(total_operations, 2);

    let rows_metric = find_metric(&finished, SK_METRIC_OPERATION_ROWS);
    let AggregatedMetrics::U64(MetricData::Histogram(rows_histogram)) = rows_metric.data() else {
        panic!("expected u64 histogram for rows");
    };
    assert_eq!(rows_histogram.data_points().count(), 2);

    let columns_metric = find_metric(&finished, SK_METRIC_OPERATION_COLUMNS);
    let AggregatedMetrics::U64(MetricData::Histogram(columns_histogram)) = columns_metric.data()
    else {
        panic!("expected u64 histogram for columns");
    };
    assert_eq!(columns_histogram.data_points().count(), 2);

    provider.shutdown().expect("meter shutdown should succeed");
}

#[test]
fn failed_operation_increments_error_counter_labelled_by_operation_and_backend() {
    use sciencekit_common::{SKError, SKOperationAttributes, SKOperationKind, sk_run_operation};

    let exporter = InMemoryMetricExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-error-metrics-test")
        .build()
        .expect("configuration should build");
    let provider = sk_build_meter_provider(&configuration, exporter.clone());
    let subscriber = tracing_subscriber::registry().with(sk_build_metrics_layer(&provider));

    let train = array![[1.0_f64, 2.0], [3.0, 4.0]];
    tracing::subscriber::with_default(subscriber, || {
        let scaler = SKStandardScaler::build()
            .backend(SKBackendKind::Faer)
            .build()
            .expect("scaler should build");
        let _fitted = scaler.fit(&train).expect("fit should succeed");

        let transform_failure = sk_run_operation(
            SKOperationAttributes {
                operation: SKOperationKind::Transform,
                rows_count: 2,
                columns_count: 3,
                execution_mode: SKExecutionMode::InProcessSynchronous,
                backend: SKBackendKind::MatrixMultiply,
            },
            || -> Result<(), SKError> { Err(SKError::shape_mismatch_2d(2, 2, 2, 3)) },
        );
        assert!(transform_failure.is_err());
    });

    provider.force_flush().expect("metric flush should succeed");
    let finished = exporter.get_finished_metrics().expect("metrics readable");
    let error_metric = find_metric(&finished, SK_METRIC_OPERATION_ERRORS);
    let AggregatedMetrics::U64(MetricData::Sum(error_sum)) = error_metric.data() else {
        panic!("expected u64 counter for operation errors");
    };
    let error_points: Vec<_> = error_sum.data_points().collect();
    assert_eq!(error_points.len(), 1);
    let transform_point = error_points
        .iter()
        .find(|point| {
            let attrs: Vec<_> = point.attributes().cloned().collect();
            attribute_map(&attrs).get("operation").map(String::as_str) == Some("transform")
        })
        .expect("transform error counter point should exist");
    assert_eq!(transform_point.value(), 1);
    let transform_attrs: Vec<_> = transform_point.attributes().cloned().collect();
    let labels = attribute_map(&transform_attrs);
    assert_eq!(
        labels.get("backend").map(String::as_str),
        Some("matrixmultiply")
    );

    provider.shutdown().expect("meter shutdown should succeed");
}
