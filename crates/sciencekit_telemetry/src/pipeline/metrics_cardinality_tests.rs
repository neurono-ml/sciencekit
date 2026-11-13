//! Companion test proving operation metric labels are bounded to the closed
//! vocabulary (spec `telemetry-export`).

use std::collections::HashMap;

use opentelemetry::Value;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use sciencekit_common::SKBuilder;
use tracing_subscriber::prelude::*;

use super::{SK_METRIC_OPERATION_COUNT, sk_build_meter_provider, sk_build_metrics_layer};
use crate::configuration::SKTelemetryConfiguration;

#[test]
fn metrics_layer_bounds_label_cardinality_to_closed_vocabulary() {
    let exporter = InMemoryMetricExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-cardinality-test")
        .build()
        .expect("configuration builds");
    let provider = sk_build_meter_provider(&configuration, exporter.clone());
    let subscriber = tracing_subscriber::registry().with(sk_build_metrics_layer(&provider));

    tracing::subscriber::with_default(subscriber, || {
        let span = tracing::info_span!(
            "sciencekit.operation",
            operation = "user-id-98765",
            backend = "unbounded-custom-backend",
            mode = "unbounded-mode",
            rows = 4_u64,
            columns = 2_u64,
        );
        let _entered = span.enter();
    });

    provider.force_flush().expect("metric flush succeeds");
    let finished = exporter.get_finished_metrics().expect("metrics readable");
    let count_metric = finished
        .iter()
        .flat_map(|resource_metrics| resource_metrics.scope_metrics())
        .flat_map(|scope_metrics| scope_metrics.metrics())
        .find(|metric| metric.name() == SK_METRIC_OPERATION_COUNT)
        .expect("operation count metric exported");
    let AggregatedMetrics::U64(MetricData::Sum(sum)) = count_metric.data() else {
        panic!("expected u64 sum");
    };
    let points: Vec<_> = sum.data_points().collect();
    let labels: HashMap<String, String> = points[0]
        .attributes()
        .map(|entry| {
            let text = match &entry.value {
                Value::String(value) => value.to_string(),
                other => format!("{other:?}"),
            };
            (entry.key.as_str().to_string(), text)
        })
        .collect();

    assert_eq!(labels.get("operation").map(String::as_str), Some("unknown"));
    assert_eq!(labels.get("backend").map(String::as_str), Some("unknown"));
    assert_eq!(labels.get("mode").map(String::as_str), Some("unknown"));
    provider.shutdown().expect("meter shutdown succeeds");
}
