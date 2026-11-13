//! Companion lifecycle tests for `init`, `force_flush`, `shutdown`,
//! double-initialization rejection and `Drop` cleanup (spec `telemetry-export`).

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use ndarray::array;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::logs::InMemoryLogExporter;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SpanData, SpanExporter};
use sciencekit_common::{SKBuilder, SKError, SKUnsupervisedFit};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::SKTelemetry;
use super::global_state::{GLOBAL_TELEMETRY_TEST_LOCK, sk_reset_global_telemetry_for_tests};
use crate::configuration::SKTelemetryConfiguration;

#[derive(Clone, Debug)]
struct LifecycleProbeSpanExporter {
    inner: InMemorySpanExporter,
    exported_count: Arc<AtomicUsize>,
    shutdown_observed: Arc<AtomicBool>,
}

impl LifecycleProbeSpanExporter {
    fn new() -> Self {
        Self {
            inner: InMemorySpanExporter::default(),
            exported_count: Arc::new(AtomicUsize::new(0)),
            shutdown_observed: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl SpanExporter for LifecycleProbeSpanExporter {
    fn export(&self, batch: Vec<SpanData>) -> impl Future<Output = OTelSdkResult> + Send {
        self.exported_count.fetch_add(batch.len(), Ordering::AcqRel);
        self.inner.export(batch)
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.shutdown_observed.store(true, Ordering::Release);
        self.inner.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

#[test]
fn force_flush_exports_pending_signals_and_shutdown_stops_further_exports() {
    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let span_exporter = LifecycleProbeSpanExporter::new();
    let log_exporter = InMemoryLogExporter::default();
    let metric_exporter = InMemoryMetricExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-lifecycle-test")
        .build()
        .expect("configuration should build");

    let telemetry = SKTelemetry::init_with_exporters(
        configuration,
        span_exporter.clone(),
        log_exporter.clone(),
        metric_exporter.clone(),
    )
    .expect("first initialization should succeed");
    assert!(SKTelemetry::is_initialized());

    let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let scaler = SKStandardScaler::build().build().expect("scaler builds");
    let _fitted = scaler.fit(&data).expect("fit succeeds");
    tracing::info!(stage = "lifecycle", "model fitted");

    telemetry.force_flush().expect("force_flush should succeed");
    assert_eq!(span_exporter.exported_count.load(Ordering::Acquire), 1);
    assert_eq!(log_exporter.get_emitted_logs().unwrap().len(), 1);
    assert!(!metric_exporter.get_finished_metrics().unwrap().is_empty());

    telemetry.shutdown().expect("shutdown should succeed");
    assert!(!SKTelemetry::is_initialized());
    assert!(span_exporter.shutdown_observed.load(Ordering::Acquire));

    let _fitted_after_shutdown = scaler.fit(&data).expect("post-shutdown fit succeeds");
    assert_eq!(span_exporter.exported_count.load(Ordering::Acquire), 1);
}

#[test]
fn double_initialization_returns_structured_error_without_replacing_pipeline() {
    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let first_configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-primary")
        .build()
        .expect("configuration builds");
    let second_configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-duplicate")
        .build()
        .expect("configuration builds");

    let telemetry = SKTelemetry::init_with_exporters(
        first_configuration,
        InMemorySpanExporter::default(),
        InMemoryLogExporter::default(),
        InMemoryMetricExporter::default(),
    )
    .expect("first init succeeds");

    let second_attempt = SKTelemetry::init_with_exporters(
        second_configuration,
        InMemorySpanExporter::default(),
        InMemoryLogExporter::default(),
        InMemoryMetricExporter::default(),
    );
    let error = match second_attempt {
        Ok(_) => panic!("second initialization must fail"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        SKError::UnsupportedOperation {
            operation: "telemetry_init",
            ..
        }
    ));
    assert_eq!(
        telemetry.configuration().service_name(),
        "sciencekit-primary"
    );

    telemetry.shutdown().expect("shutdown succeeds");
}

#[test]
fn dropping_telemetry_guard_flushes_pending_signals_and_detaches_safely() {
    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let span_exporter = LifecycleProbeSpanExporter::new();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-drop-test")
        .build()
        .expect("configuration builds");

    {
        let _telemetry = SKTelemetry::init_with_exporters(
            configuration,
            span_exporter.clone(),
            InMemoryLogExporter::default(),
            InMemoryMetricExporter::default(),
        )
        .expect("init succeeds");
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let scaler = SKStandardScaler::build().build().expect("scaler builds");
        let _ = scaler.fit(&data).expect("fit succeeds");
    }

    assert!(!SKTelemetry::is_initialized());
    assert_eq!(span_exporter.exported_count.load(Ordering::Acquire), 1);
    assert!(span_exporter.shutdown_observed.load(Ordering::Acquire));
}
