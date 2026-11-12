//! Validated OpenTelemetry export configuration (spec `telemetry-export`).

use std::collections::BTreeMap;
use std::time::Duration;

use sciencekit_common::builders::SKBuilderState;
use sciencekit_common::{SKBackendKind, SKExecutionMode};

use super::builder::SKTelemetryConfigurationBuilder;
use super::{
    SKTelemetryBatchConfiguration, SKTelemetryResource, SKTelemetrySampler, SKTelemetryTransport,
};

/// Validated configuration driving the trace, log and metric export pipelines.
#[derive(Debug, Clone, PartialEq)]
pub struct SKTelemetryConfiguration {
    pub(crate) resource: SKTelemetryResource,
    pub(crate) endpoint: String,
    pub(crate) transport: SKTelemetryTransport,
    pub(crate) headers: BTreeMap<String, String>,
    pub(crate) timeout: Duration,
    pub(crate) flush_timeout: Duration,
    pub(crate) sampler: SKTelemetrySampler,
    pub(crate) batch: SKTelemetryBatchConfiguration,
    pub(crate) backend: SKBackendKind,
    pub(crate) state: SKBuilderState,
}

impl SKTelemetryConfiguration {
    /// Open a new [`SKTelemetryConfigurationBuilder`].
    pub fn builder() -> SKTelemetryConfigurationBuilder {
        SKTelemetryConfigurationBuilder::new()
    }

    /// Alias matching the workspace estimator builder entry point.
    pub fn build() -> SKTelemetryConfigurationBuilder {
        SKTelemetryConfigurationBuilder::new()
    }

    /// Resource metadata attached to every exported trace, log and metric.
    pub fn resource(&self) -> &SKTelemetryResource {
        &self.resource
    }

    /// Configured `service.name` resource attribute.
    pub fn service_name(&self) -> &str {
        self.resource.service_name()
    }

    /// Optional `service.version` resource attribute.
    pub fn service_version(&self) -> Option<&str> {
        self.resource.service_version()
    }

    /// Optional `deployment.environment.name` resource attribute.
    pub fn deployment_environment(&self) -> Option<&str> {
        self.resource.deployment_environment()
    }

    /// Validated OTLP collector endpoint URL.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Selected OTLP wire transport (`Grpc` or `Http`).
    pub fn transport(&self) -> SKTelemetryTransport {
        self.transport
    }

    /// Request headers / metadata pairs sent to the OTLP collector.
    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    /// Per-request OTLP export timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Timeout applied to lifecycle `force_flush` and `shutdown` operations.
    pub fn flush_timeout(&self) -> Duration {
        self.flush_timeout
    }

    /// Trace sampling strategy.
    pub fn sampler(&self) -> SKTelemetrySampler {
        self.sampler
    }

    /// Batch queue and periodic metric export parameters.
    pub fn batch(&self) -> &SKTelemetryBatchConfiguration {
        &self.batch
    }

    /// Configured math backend recorded for builder uniformity.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Configured execution mode intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.state.execution_intent()
    }
}
