//! Companion unit tests for telemetry configuration, resource and sampler types.

use std::collections::BTreeMap;
use std::time::Duration;

use opentelemetry::{Key, Value};
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode};

use super::{
    SKTelemetryBatchConfiguration, SKTelemetryConfiguration, SKTelemetryResource,
    SKTelemetrySampler, SKTelemetryTransport,
};

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

#[test]
fn configuration_builder_constructs_complete_configuration_with_defaults_and_overrides() {
    let default_http = SKTelemetryConfiguration::builder()
        .service_name("sciencekit")
        .transport(SKTelemetryTransport::Http)
        .build()
        .expect("valid HTTP configuration should build");
    assert_eq!(default_http.endpoint(), "http://localhost:4318");
    assert_eq!(default_http.transport(), SKTelemetryTransport::Http);
    assert_eq!(default_http.backend(), SKBackendKind::Faer);
    assert_eq!(default_http.execution_mode(), SKExecutionMode::Automatic);

    let batch = SKTelemetryBatchConfiguration::builder()
        .maximum_queue_size(1024)
        .maximum_export_batch_size(256)
        .scheduled_delay(Duration::from_millis(250))
        .export_timeout(Duration::from_secs(15))
        .periodic_metric_interval(Duration::from_secs(10))
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("valid batch configuration should build");

    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-trainer")
        .service_version("0.7.0")
        .deployment_environment("staging")
        .resource_attribute("cluster.name", "gpu-pool-a")
        .endpoint("https://otel.example.internal:4317")
        .transport(SKTelemetryTransport::Grpc)
        .header("x-api-key", "secret-token")
        .headers([("x-tenant", "research")])
        .timeout(Duration::from_secs(12))
        .flush_timeout(Duration::from_secs(4))
        .sampler(SKTelemetrySampler::ParentBased(0.1))
        .batch(batch)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("complete configuration should build");

    assert_eq!(configuration.service_name(), "sciencekit-trainer");
    assert_eq!(configuration.service_version(), Some("0.7.0"));
    assert_eq!(configuration.deployment_environment(), Some("staging"));
    assert_eq!(
        configuration.endpoint(),
        "https://otel.example.internal:4317"
    );
    assert_eq!(configuration.headers().len(), 2);
    assert_eq!(configuration.timeout(), Duration::from_secs(12));
    assert_eq!(configuration.flush_timeout(), Duration::from_secs(4));
    assert_eq!(
        configuration.sampler(),
        SKTelemetrySampler::ParentBased(0.1)
    );
    assert_eq!(configuration.batch().maximum_queue_size(), 1024);
    assert_eq!(configuration.batch().maximum_export_batch_size(), 256);
    assert_eq!(configuration.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        configuration.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}
