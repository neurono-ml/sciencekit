//! Builder for [`super::SKTelemetryConfiguration`] (spec `telemetry-export`).

use std::collections::BTreeMap;
use std::time::Duration;

use sciencekit_common::builders::{SKBuilder, SKBuilderState};
use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode};

use super::validation::{
    sk_validate_endpoint, sk_validate_headers, sk_validate_optional_text,
    sk_validate_positive_duration, sk_validate_resource_attributes, sk_validate_service_name,
};
use super::{
    SKTelemetryBatchConfiguration, SKTelemetryConfiguration, SKTelemetryResource,
    SKTelemetrySampler, SKTelemetryTransport,
};

/// Builder constructing a validated [`SKTelemetryConfiguration`].
#[derive(Debug, Clone)]
pub struct SKTelemetryConfigurationBuilder {
    service_name: Option<String>,
    service_version: Option<String>,
    deployment_environment: Option<String>,
    resource_attributes: BTreeMap<String, String>,
    endpoint: Option<String>,
    transport: SKTelemetryTransport,
    headers: BTreeMap<String, String>,
    timeout: Duration,
    flush_timeout: Duration,
    sampler: SKTelemetrySampler,
    batch: SKTelemetryBatchConfiguration,
    backend: SKBackendKind,
    state: SKBuilderState,
}

impl SKTelemetryConfigurationBuilder {
    /// Start a builder with gRPC transport, `AlwaysOn` sampling, `Faer` backend and `Automatic` mode.
    pub fn new() -> Self {
        Self {
            service_name: None,
            service_version: None,
            deployment_environment: None,
            resource_attributes: BTreeMap::new(),
            endpoint: None,
            transport: SKTelemetryTransport::Grpc,
            headers: BTreeMap::new(),
            timeout: Duration::from_secs(10),
            flush_timeout: Duration::from_secs(5),
            sampler: SKTelemetrySampler::AlwaysOn,
            batch: SKTelemetryBatchConfiguration::default_configuration(),
            backend: SKBackendKind::Faer,
            state: SKBuilderState::new(),
        }
    }

    /// Set the required `service.name` resource attribute.
    pub fn service_name(mut self, service_name: impl Into<String>) -> Self {
        self.service_name = Some(service_name.into());
        self
    }

    /// Set the optional `service.version` resource attribute.
    pub fn service_version(mut self, service_version: impl Into<String>) -> Self {
        self.service_version = Some(service_version.into());
        self
    }

    /// Set the optional `deployment.environment.name` resource attribute.
    pub fn deployment_environment(mut self, deployment_environment: impl Into<String>) -> Self {
        self.deployment_environment = Some(deployment_environment.into());
        self
    }

    /// Insert an additional resource attribute key-value pair.
    pub fn resource_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.resource_attributes.insert(key.into(), value.into());
        self
    }

    /// Override the OTLP collector endpoint URL; defaults to the transport's standard endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Select the OTLP wire transport (`Grpc` or `Http`).
    pub fn transport(mut self, transport: SKTelemetryTransport) -> Self {
        self.transport = transport;
        self
    }

    /// Insert one request header / metadata entry sent with OTLP exports.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Insert multiple request headers / metadata entries.
    pub fn headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers.extend(
            headers
                .into_iter()
                .map(|(name, value)| (name.into(), value.into())),
        );
        self
    }

    /// Set the per-request OTLP export timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the lifecycle `force_flush` and `shutdown` timeout.
    pub fn flush_timeout(mut self, flush_timeout: Duration) -> Self {
        self.flush_timeout = flush_timeout;
        self
    }

    /// Select the trace sampling strategy.
    pub fn sampler(mut self, sampler: SKTelemetrySampler) -> Self {
        self.sampler = sampler;
        self
    }

    /// Configure batch export queue sizes and intervals.
    pub fn batch(mut self, batch: SKTelemetryBatchConfiguration) -> Self {
        self.batch = batch;
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

impl Default for SKTelemetryConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKTelemetryConfiguration> for SKTelemetryConfigurationBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKTelemetryConfiguration, SKError> {
        let service_name = sk_validate_service_name(self.service_name.as_deref())?;
        let service_version = sk_validate_optional_text("service_version", self.service_version)?;
        let deployment_environment =
            sk_validate_optional_text("deployment_environment", self.deployment_environment)?;
        sk_validate_resource_attributes(&self.resource_attributes)?;
        let endpoint = self
            .endpoint
            .unwrap_or_else(|| self.transport.default_endpoint().to_string());
        sk_validate_endpoint(&endpoint)?;
        sk_validate_headers(&self.headers)?;
        sk_validate_positive_duration("timeout", self.timeout)?;
        sk_validate_positive_duration("flush_timeout", self.flush_timeout)?;
        self.sampler.validate()?;
        Ok(SKTelemetryConfiguration {
            resource: SKTelemetryResource::from_parts(
                service_name,
                service_version,
                deployment_environment,
                self.resource_attributes,
            ),
            endpoint,
            transport: self.transport,
            headers: self.headers,
            timeout: self.timeout,
            flush_timeout: self.flush_timeout,
            sampler: self.sampler,
            batch: self.batch,
            backend: self.backend,
            state: self.state,
        })
    }
}
