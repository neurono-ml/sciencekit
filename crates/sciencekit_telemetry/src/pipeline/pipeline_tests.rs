//! Companion pipeline tests for resource attributes, samplers and batching
//! (spec `telemetry-export`).

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use opentelemetry::{Key, Value};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::InMemoryLogExporter;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SpanData, SpanExporter};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::SKTelemetryPipeline;
use crate::configuration::{
    SKTelemetryBatchConfiguration, SKTelemetryConfiguration, SKTelemetrySampler,
};

#[derive(Clone, Debug)]
struct ResourceCapturingSpanExporter {
    inner: InMemorySpanExporter,
    captured_resource: Arc<Mutex<Option<Resource>>>,
}

impl ResourceCapturingSpanExporter {
    fn new() -> Self {
        Self {
            inner: InMemorySpanExporter::default(),
            captured_resource: Arc::new(Mutex::new(None)),
        }
    }
}

impl SpanExporter for ResourceCapturingSpanExporter {
    fn export(&self, batch: Vec<SpanData>) -> impl Future<Output = OTelSdkResult> + Send {
        self.inner.export(batch)
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        *self.captured_resource.lock().unwrap() = Some(resource.clone());
        self.inner.set_resource(resource);
    }
}

fn assert_expected_resource(resource: &Resource) {
    assert_eq!(
        resource.get(&Key::from_static_str("service.name")),
        Some(Value::from("sciencekit-pipeline-service"))
    );
    assert_eq!(
        resource.get(&Key::from_static_str("service.version")),
        Some(Value::from("0.7.0"))
    );
    assert_eq!(
        resource.get(&Key::from_static_str("deployment.environment.name")),
        Some(Value::from("production"))
    );
    assert_eq!(
        resource.get(&Key::from_static_str("cluster.id")),
        Some(Value::from("alpha"))
    );
}

#[test]
fn configured_resource_attributes_propagate_to_traces_logs_and_metrics() {
    let span_exporter = ResourceCapturingSpanExporter::new();
    let log_exporter = InMemoryLogExporter::default();
    let metric_exporter = InMemoryMetricExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-pipeline-service")
        .service_version("0.7.0")
        .deployment_environment("production")
        .resource_attribute("cluster.id", "alpha")
        .build()
        .expect("configuration should build");
    let pipeline = SKTelemetryPipeline::from_exporters(
        configuration,
        span_exporter.clone(),
        log_exporter.clone(),
        metric_exporter.clone(),
    );

    tracing::subscriber::with_default(pipeline.subscriber(), || {
        let _ = sk_run_operation(
            SKOperationAttributes {
                operation: SKOperationKind::Fit,
                rows_count: 16,
                columns_count: 4,
                execution_mode: SKExecutionMode::InProcessSynchronous,
                backend: SKBackendKind::Faer,
            },
            || -> Result<(), SKError> {
                tracing::info!(step = "fit", "fitting model");
                Ok(())
            },
        );
    });

    pipeline
        .force_flush()
        .expect("pipeline flush should succeed");
    let trace_resource = span_exporter
        .captured_resource
        .lock()
        .unwrap()
        .clone()
        .expect("trace exporter must receive resource");
    assert_expected_resource(&trace_resource);
    assert_expected_resource(
        log_exporter.get_emitted_logs().unwrap()[0]
            .resource
            .as_ref(),
    );
    assert_expected_resource(metric_exporter.get_finished_metrics().unwrap()[0].resource());
    pipeline
        .shutdown()
        .expect("pipeline shutdown should succeed");
}

#[test]
fn sampler_and_batch_settings_govern_exported_spans_and_batches() {
    let off_spans = InMemorySpanExporter::default();
    let off_configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-sampler-off")
        .sampler(SKTelemetrySampler::AlwaysOff)
        .build()
        .expect("configuration should build");
    let off_pipeline = SKTelemetryPipeline::from_exporters(
        off_configuration,
        off_spans.clone(),
        InMemoryLogExporter::default(),
        InMemoryMetricExporter::default(),
    );
    tracing::subscriber::with_default(off_pipeline.subscriber(), || {
        let _ = sk_run_operation(
            SKOperationAttributes {
                operation: SKOperationKind::Fit,
                rows_count: 10,
                columns_count: 2,
                execution_mode: SKExecutionMode::InProcessSynchronous,
                backend: SKBackendKind::Faer,
            },
            || Ok::<(), SKError>(()),
        );
    });
    off_pipeline.force_flush().expect("flush should succeed");
    assert!(off_spans.get_finished_spans().unwrap().is_empty());
    off_pipeline.shutdown().expect("shutdown should succeed");

    let batch_spans = InMemorySpanExporter::default();
    let batch_settings = SKTelemetryBatchConfiguration::builder()
        .maximum_queue_size(64)
        .maximum_export_batch_size(4)
        .scheduled_delay(Duration::from_millis(50))
        .periodic_metric_interval(Duration::from_millis(50))
        .build()
        .expect("batch settings should build");
    let batch_configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-batch-test")
        .sampler(SKTelemetrySampler::ParentBased(1.0))
        .batch(batch_settings)
        .build()
        .expect("configuration should build");
    let batch_pipeline = SKTelemetryPipeline::from_exporters(
        batch_configuration,
        batch_spans.clone(),
        InMemoryLogExporter::default(),
        InMemoryMetricExporter::default(),
    );
    tracing::subscriber::with_default(batch_pipeline.subscriber(), || {
        (0..10_usize).for_each(|rows| {
            let _ = sk_run_operation(
                SKOperationAttributes {
                    operation: SKOperationKind::Score,
                    rows_count: rows + 1,
                    columns_count: 3,
                    execution_mode: SKExecutionMode::InProcessSynchronous,
                    backend: SKBackendKind::Faer,
                },
                || Ok::<(), SKError>(()),
            );
        });
    });
    batch_pipeline.force_flush().expect("flush should succeed");
    assert_eq!(batch_spans.get_finished_spans().unwrap().len(), 10);
    batch_pipeline.shutdown().expect("shutdown should succeed");
}
