//! Unified three-signal OpenTelemetry pipeline (spec `telemetry-export`).

use opentelemetry_sdk::logs::{LogExporter, SdkLoggerProvider};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use sciencekit_common::SKError;
use tracing_subscriber::prelude::*;

use super::{
    sk_build_log_layer, sk_build_logger_provider, sk_build_meter_provider, sk_build_metrics_layer,
    sk_build_trace_layer, sk_build_tracer_provider,
};
use crate::configuration::SKTelemetryConfiguration;

/// Bundled OpenTelemetry trace, log and metric providers built from one
/// [`SKTelemetryConfiguration`].
pub struct SKTelemetryPipeline {
    configuration: SKTelemetryConfiguration,
    tracer_provider: SdkTracerProvider,
    logger_provider: SdkLoggerProvider,
    meter_provider: SdkMeterProvider,
}

impl SKTelemetryPipeline {
    /// Construct all three signal providers over caller-supplied exporters.
    pub fn from_exporters<S, L, M>(
        configuration: SKTelemetryConfiguration,
        span_exporter: S,
        log_exporter: L,
        metric_exporter: M,
    ) -> Self
    where
        S: SpanExporter + 'static,
        L: LogExporter + 'static,
        M: PushMetricExporter + 'static,
    {
        let meter_provider = sk_build_meter_provider(&configuration, metric_exporter);
        let logger_provider = sk_build_logger_provider(&configuration, log_exporter);
        let tracer_provider = sk_build_tracer_provider(&configuration, span_exporter);
        Self {
            configuration,
            tracer_provider,
            logger_provider,
            meter_provider,
        }
    }

    /// Access the validated configuration driving this pipeline.
    pub fn configuration(&self) -> &SKTelemetryConfiguration {
        &self.configuration
    }

    /// Access the underlying [`SdkTracerProvider`].
    pub fn tracer_provider(&self) -> &SdkTracerProvider {
        &self.tracer_provider
    }

    /// Access the underlying [`SdkLoggerProvider`].
    pub fn logger_provider(&self) -> &SdkLoggerProvider {
        &self.logger_provider
    }

    /// Access the underlying [`SdkMeterProvider`].
    pub fn meter_provider(&self) -> &SdkMeterProvider {
        &self.meter_provider
    }

    /// Build a composite `tracing::Subscriber` bridging traces, logs and metrics.
    pub fn subscriber(&self) -> Box<dyn tracing::Subscriber + Send + Sync + 'static> {
        Box::new(
            tracing_subscriber::registry()
                .with(sk_build_trace_layer(&self.tracer_provider))
                .with(sk_build_log_layer(&self.logger_provider))
                .with(sk_build_metrics_layer(&self.meter_provider)),
        )
    }

    /// Flush pending traces, logs and metrics across all three providers.
    pub fn force_flush(&self) -> Result<(), SKError> {
        let mut failures = Vec::new();
        if let Err(error) = self.tracer_provider.force_flush() {
            failures.push(format!("traces: {error}"));
        }
        if let Err(error) = self.logger_provider.force_flush() {
            failures.push(format!("logs: {error}"));
        }
        if let Err(error) = self.meter_provider.force_flush() {
            failures.push(format!("metrics: {error}"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(SKError::UnsupportedOperation {
                operation: "telemetry_force_flush",
                reason: failures.join("; "),
            })
        }
    }

    /// Flush and shut down all three providers within the configured `flush_timeout`.
    pub fn shutdown(&self) -> Result<(), SKError> {
        let timeout = self.configuration.flush_timeout();
        let _ = self.force_flush();
        let mut failures = Vec::new();
        if let Err(error) = self.tracer_provider.shutdown_with_timeout(timeout) {
            failures.push(format!("traces: {error}"));
        }
        if let Err(error) = self.logger_provider.shutdown_with_timeout(timeout) {
            failures.push(format!("logs: {error}"));
        }
        if let Err(error) = self.meter_provider.shutdown_with_timeout(timeout) {
            failures.push(format!("metrics: {error}"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(SKError::UnsupportedOperation {
                operation: "telemetry_shutdown",
                reason: failures.join("; "),
            })
        }
    }
}
