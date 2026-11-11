//! Global telemetry handle and initialization entry points
//! (spec `telemetry-export`).

use std::sync::atomic::AtomicBool;

use opentelemetry::global;
use opentelemetry_sdk::logs::LogExporter;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::trace::SpanExporter;
use sciencekit_common::SKError;

use super::global_state::{
    sk_active_exporter_count, sk_claim_initialization_slot, sk_install_global_subscriber,
    sk_is_active_initialized, sk_release_failed_initialization_slot,
};
use crate::configuration::SKTelemetryConfiguration;
use crate::pipeline::{SKTelemetryPipeline, sk_build_otlp_pipeline};

/// Process-wide OpenTelemetry pipeline handle managing global registration,
/// deterministic flushing and clean shutdown.
pub struct SKTelemetry {
    pub(super) pipeline: SKTelemetryPipeline,
    pub(super) is_shut_down: AtomicBool,
}

impl SKTelemetry {
    /// Initialize the global telemetry pipeline over OTLP network exporters.
    ///
    /// Fails with [`SKError::UnsupportedOperation`] if a global pipeline has
    /// already been initialized in this process or if the selected transport's
    /// Cargo feature (`otlp-grpc` or `otlp-http`) is disabled.
    pub fn init(configuration: SKTelemetryConfiguration) -> Result<Self, SKError> {
        sk_claim_initialization_slot()?;
        let pipeline = match sk_build_otlp_pipeline(configuration) {
            Ok(built) => built,
            Err(error) => {
                sk_release_failed_initialization_slot();
                return Err(error);
            }
        };
        Self::activate_pipeline(pipeline)
    }

    /// Initialize the global telemetry pipeline over caller-supplied exporters.
    ///
    /// Shares the exact global registration, `force_flush`, `shutdown` and
    /// `Drop` lifecycle of [`SKTelemetry::init`].
    pub fn init_with_exporters<S, L, M>(
        configuration: SKTelemetryConfiguration,
        span_exporter: S,
        log_exporter: L,
        metric_exporter: M,
    ) -> Result<Self, SKError>
    where
        S: SpanExporter + 'static,
        L: LogExporter + 'static,
        M: PushMetricExporter + 'static,
    {
        sk_claim_initialization_slot()?;
        let pipeline = SKTelemetryPipeline::from_exporters(
            configuration,
            span_exporter,
            log_exporter,
            metric_exporter,
        );
        Self::activate_pipeline(pipeline)
    }

    fn activate_pipeline(pipeline: SKTelemetryPipeline) -> Result<Self, SKError> {
        if let Err(error) = sk_install_global_subscriber(pipeline.subscriber()) {
            sk_release_failed_initialization_slot();
            return Err(error);
        }
        global::set_meter_provider(pipeline.meter_provider().clone());
        global::set_tracer_provider(pipeline.tracer_provider().clone());
        Ok(Self {
            pipeline,
            is_shut_down: AtomicBool::new(false),
        })
    }

    /// Whether an active global telemetry pipeline is currently installed.
    pub fn is_initialized() -> bool {
        sk_is_active_initialized()
    }

    /// Number of active global telemetry exporters currently installed (`0` when uninitialized).
    pub fn active_exporter_count() -> usize {
        sk_active_exporter_count()
    }

    /// Access the validated configuration of the active pipeline.
    pub fn configuration(&self) -> &SKTelemetryConfiguration {
        self.pipeline.configuration()
    }
}
