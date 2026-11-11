//! Companion unit tests for telemetry configuration, resource and sampler types.

use std::collections::BTreeMap;

use opentelemetry::{Key, Value};

use super::{SKTelemetryResource, SKTelemetrySampler, SKTelemetryTransport};

#[test]
fn transport_defaults_to_grpc_and_exposes_standard_endpoints() {
    assert_eq!(SKTelemetryTransport::default(), SKTelemetryTransport::Grpc);
    assert_eq!(SKTelemetryTransport::Grpc.name(), "grpc");
    assert_eq!(
        SKTelemetryTransport::Grpc.default_endpoint(),
        "http://localhost:4317"
    );
    assert_eq!(SKTelemetryTransport::Http.name(), "http");
    assert_eq!(
        SKTelemetryTransport::Http.default_endpoint(),
        "http://localhost:4318"
    );
    assert_eq!(SKTelemetryTransport::Grpc.to_string(), "grpc");
    assert_eq!(SKTelemetryTransport::Http.to_string(), "http");
}

#[test]
fn resource_projects_service_identity_and_custom_attributes_into_sdk_resource() {
    let mut custom_attributes = BTreeMap::new();
    custom_attributes.insert("host.region".to_string(), "eu-west-1".to_string());

    let resource = SKTelemetryResource::from_parts(
        "sciencekit-worker".to_string(),
        Some("0.7.0".to_string()),
        Some("production".to_string()),
        custom_attributes,
    );

    assert_eq!(resource.service_name(), "sciencekit-worker");
    assert_eq!(resource.service_version(), Some("0.7.0"));
    assert_eq!(resource.deployment_environment(), Some("production"));

    let sdk_resource = resource.to_sdk_resource();
    assert_eq!(
        sdk_resource.get(&Key::from_static_str("service.name")),
        Some(Value::from("sciencekit-worker"))
    );
    assert_eq!(
        sdk_resource.get(&Key::from_static_str("service.version")),
        Some(Value::from("0.7.0"))
    );
    assert_eq!(
        sdk_resource.get(&Key::from_static_str("deployment.environment.name")),
        Some(Value::from("production"))
    );
    assert_eq!(
        sdk_resource.get(&Key::from_static_str("host.region")),
        Some(Value::from("eu-west-1"))
    );
}

#[test]
fn sampler_validates_ratios_and_builds_sdk_samplers() {
    assert_eq!(SKTelemetrySampler::default(), SKTelemetrySampler::AlwaysOn);
    assert!(SKTelemetrySampler::AlwaysOn.validate().is_ok());
    assert!(SKTelemetrySampler::AlwaysOff.validate().is_ok());
    assert!(
        SKTelemetrySampler::TraceIdRatioBased(0.0)
            .validate()
            .is_ok()
    );
    assert!(SKTelemetrySampler::ParentBased(0.25).validate().is_ok());
    assert!(SKTelemetrySampler::ParentBased(1.0).validate().is_ok());

    assert!(
        SKTelemetrySampler::TraceIdRatioBased(-0.1)
            .validate()
            .is_err()
    );
    assert!(SKTelemetrySampler::ParentBased(1.1).validate().is_err());
    assert!(
        SKTelemetrySampler::ParentBased(f64::NAN)
            .validate()
            .is_err()
    );

    let _ = SKTelemetrySampler::ParentBased(0.5).to_sdk_sampler();
}
