//! Builder for [`super::SKTelemetryBatchConfiguration`] (spec `telemetry-export`).

use std::time::Duration;

use sciencekit_common::builders::{SKBuilder, SKBuilderState, sk_validate_hyperparameter};
use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode};

use super::SKTelemetryBatchConfiguration;

/// Builder for [`SKTelemetryBatchConfiguration`].
#[derive(Debug, Clone, Copy)]
pub struct SKTelemetryBatchConfigurationBuilder {
    maximum_queue_size: usize,
    maximum_export_batch_size: usize,
    scheduled_delay: Duration,
    export_timeout: Duration,
    periodic_metric_interval: Duration,
    backend: SKBackendKind,
    state: SKBuilderState,
}

impl SKTelemetryBatchConfigurationBuilder {
    /// Start a builder with OpenTelemetry specification defaults.
    pub fn new() -> Self {
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

    /// Set the maximum buffered queue capacity (must be at least `1`).
    pub fn maximum_queue_size(mut self, maximum_queue_size: usize) -> Self {
        self.maximum_queue_size = maximum_queue_size;
        self
    }

    /// Set the maximum batch size per export (must lie in `1..=maximum_queue_size`).
    pub fn maximum_export_batch_size(mut self, maximum_export_batch_size: usize) -> Self {
        self.maximum_export_batch_size = maximum_export_batch_size;
        self
    }

    /// Set the interval between background batch flushes.
    pub fn scheduled_delay(mut self, scheduled_delay: Duration) -> Self {
        self.scheduled_delay = scheduled_delay;
        self
    }

    /// Set the per-batch export timeout.
    pub fn export_timeout(mut self, export_timeout: Duration) -> Self {
        self.export_timeout = export_timeout;
        self
    }

    /// Set the periodic metric reader export interval.
    pub fn periodic_metric_interval(mut self, periodic_metric_interval: Duration) -> Self {
        self.periodic_metric_interval = periodic_metric_interval;
        self
    }

    /// Select the math backend recorded for builder uniformity; defaults to [`SKBackendKind::Faer`].
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Override the execution mode intent; defaults to [`SKExecutionMode::Automatic`].
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl Default for SKTelemetryBatchConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKTelemetryBatchConfiguration> for SKTelemetryBatchConfigurationBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKTelemetryBatchConfiguration, SKError> {
        sk_validate_hyperparameter(
            "maximum_queue_size",
            self.maximum_queue_size >= 1,
            "maximum queue size must be at least 1",
        )?;
        sk_validate_hyperparameter(
            "maximum_export_batch_size",
            self.maximum_export_batch_size >= 1
                && self.maximum_export_batch_size <= self.maximum_queue_size,
            "maximum export batch size must lie in 1..=maximum_queue_size",
        )?;
        sk_validate_hyperparameter(
            "scheduled_delay",
            !self.scheduled_delay.is_zero(),
            "scheduled delay must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "export_timeout",
            !self.export_timeout.is_zero(),
            "export timeout must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "periodic_metric_interval",
            !self.periodic_metric_interval.is_zero(),
            "periodic metric interval must be strictly positive",
        )?;
        Ok(SKTelemetryBatchConfiguration {
            maximum_queue_size: self.maximum_queue_size,
            maximum_export_batch_size: self.maximum_export_batch_size,
            scheduled_delay: self.scheduled_delay,
            export_timeout: self.export_timeout,
            periodic_metric_interval: self.periodic_metric_interval,
            backend: self.backend,
            state: self.state,
        })
    }
}
