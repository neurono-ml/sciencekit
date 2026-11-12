//! Batch export queue and periodic reader configuration (spec `telemetry-export`).

use std::time::Duration;

use opentelemetry_sdk::logs::{
    BatchConfig as LogBatchConfig, BatchConfigBuilder as LogBatchBuilder,
};
use opentelemetry_sdk::trace::{
    BatchConfig as SpanBatchConfig, BatchConfigBuilder as SpanBatchBuilder,
};
use sciencekit_common::builders::SKBuilderState;
use sciencekit_common::{SKBackendKind, SKExecutionMode};

use super::SKTelemetryBatchConfigurationBuilder;

/// Validated batching parameters shared by trace, log and metric exporters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKTelemetryBatchConfiguration {
    pub(super) maximum_queue_size: usize,
    pub(super) maximum_export_batch_size: usize,
    pub(super) scheduled_delay: Duration,
    pub(super) export_timeout: Duration,
    pub(super) periodic_metric_interval: Duration,
    pub(super) backend: SKBackendKind,
    pub(super) state: SKBuilderState,
}

impl SKTelemetryBatchConfiguration {
    /// Open a builder with OpenTelemetry specification defaults.
    pub fn builder() -> SKTelemetryBatchConfigurationBuilder {
        SKTelemetryBatchConfigurationBuilder::new()
    }

    /// Alias matching the workspace estimator builder entry point.
    pub fn build() -> SKTelemetryBatchConfigurationBuilder {
        SKTelemetryBatchConfigurationBuilder::new()
    }

    pub(crate) fn default_configuration() -> Self {
        Self {
            maximum_queue_size: 2048,
            maximum_export_batch_size: 512,
            scheduled_delay: Duration::from_millis(5000),
            export_timeout: Duration::from_secs(30),
            periodic_metric_interval: Duration::from_secs(60),
            backend: SKBackendKind::Faer,
            state: SKBuilderState::new(),
        }
    }

    /// Maximum buffered spans or log records before dropping.
    pub fn maximum_queue_size(&self) -> usize {
        self.maximum_queue_size
    }

    /// Maximum items exported in a single batch.
    pub fn maximum_export_batch_size(&self) -> usize {
        self.maximum_export_batch_size
    }

    /// Delay between scheduled background batch exports.
    pub fn scheduled_delay(&self) -> Duration {
        self.scheduled_delay
    }

    /// Maximum duration allowed for one batch export call.
    pub fn export_timeout(&self) -> Duration {
        self.export_timeout
    }

    /// Interval between periodic metric reader collections.
    pub fn periodic_metric_interval(&self) -> Duration {
        self.periodic_metric_interval
    }

    /// Configured math backend recorded for builder uniformity.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Configured execution mode intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.state.execution_intent()
    }

    /// Build the SDK trace [`SpanBatchConfig`].
    pub fn to_sdk_trace_batch_configuration(&self) -> SpanBatchConfig {
        SpanBatchBuilder::default()
            .with_max_queue_size(self.maximum_queue_size)
            .with_max_export_batch_size(self.maximum_export_batch_size)
            .with_scheduled_delay(self.scheduled_delay)
            .build()
    }

    /// Build the SDK log [`LogBatchConfig`].
    pub fn to_sdk_log_batch_configuration(&self) -> LogBatchConfig {
        LogBatchBuilder::default()
            .with_max_queue_size(self.maximum_queue_size)
            .with_max_export_batch_size(self.maximum_export_batch_size)
            .with_scheduled_delay(self.scheduled_delay)
            .build()
    }
}
