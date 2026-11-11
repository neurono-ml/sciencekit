//! OpenTelemetry resource attributes describing the producing service
//! (spec `telemetry-export`).

use std::collections::BTreeMap;

use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;

/// Validated OpenTelemetry resource metadata attached to every exported signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKTelemetryResource {
    pub(crate) service_name: String,
    pub(crate) service_version: Option<String>,
    pub(crate) deployment_environment: Option<String>,
    pub(crate) attributes: BTreeMap<String, String>,
}

impl SKTelemetryResource {
    /// Construct resource metadata from validated fields.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_parts(
        service_name: String,
        service_version: Option<String>,
        deployment_environment: Option<String>,
        attributes: BTreeMap<String, String>,
    ) -> Self {
        Self {
            service_name,
            service_version,
            deployment_environment,
            attributes,
        }
    }

    /// Configured `service.name` resource attribute.
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Optional `service.version` resource attribute.
    pub fn service_version(&self) -> Option<&str> {
        self.service_version.as_deref()
    }

    /// Optional `deployment.environment.name` resource attribute.
    pub fn deployment_environment(&self) -> Option<&str> {
        self.deployment_environment.as_deref()
    }

    /// Additional custom resource attributes in deterministic key order.
    pub fn attributes(&self) -> &BTreeMap<String, String> {
        &self.attributes
    }

    /// Build the corresponding [`opentelemetry_sdk::Resource`].
    pub fn to_sdk_resource(&self) -> Resource {
        let mut attributes = Vec::with_capacity(self.attributes.len() + 2);
        if let Some(version) = &self.service_version {
            attributes.push(KeyValue::new("service.version", version.clone()));
        }
        if let Some(environment) = &self.deployment_environment {
            attributes.push(KeyValue::new(
                "deployment.environment.name",
                environment.clone(),
            ));
        }
        attributes.extend(
            self.attributes
                .iter()
                .map(|(key, value)| KeyValue::new(key.clone(), value.clone())),
        );
        Resource::builder()
            .with_service_name(self.service_name.clone())
            .with_attributes(attributes)
            .build()
    }
}
