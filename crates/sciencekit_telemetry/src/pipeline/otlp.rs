//! OTLP network exporter construction behind `otlp-grpc` and `otlp-http`
//! (spec `telemetry-export`).

use sciencekit_common::SKError;

use super::SKTelemetryPipeline;
use crate::configuration::{SKTelemetryConfiguration, SKTelemetryTransport};

/// Build an [`SKTelemetryPipeline`] backed by OTLP network exporters matching
/// `configuration.transport()`.
pub fn sk_build_otlp_pipeline(
    configuration: SKTelemetryConfiguration,
) -> Result<SKTelemetryPipeline, SKError> {
    match configuration.transport() {
        SKTelemetryTransport::Grpc => build_grpc_pipeline(configuration),
        SKTelemetryTransport::Http => build_http_pipeline(configuration),
    }
}

#[cfg(feature = "otlp-grpc")]
fn build_grpc_pipeline(
    configuration: SKTelemetryConfiguration,
) -> Result<SKTelemetryPipeline, SKError> {
    use http::{HeaderMap, HeaderName, HeaderValue};
    use opentelemetry_otlp::tonic_types::metadata::MetadataMap;
    use opentelemetry_otlp::{
        LogExporter, MetricExporter, SpanExporter, WithExportConfig, WithTonicConfig,
    };

    let endpoint = configuration.endpoint().to_string();
    let timeout = configuration.timeout();
    let mut header_map = HeaderMap::with_capacity(configuration.headers().len());
    for (name, value) in configuration.headers() {
        let header_name =
            HeaderName::from_bytes(name.to_ascii_lowercase().as_bytes()).map_err(|error| {
                SKError::InvalidHyperparameter {
                    name: "headers",
                    reason: format!("invalid gRPC metadata key `{name}`: {error}"),
                }
            })?;
        let header_value =
            HeaderValue::from_str(value).map_err(|error| SKError::InvalidHyperparameter {
                name: "headers",
                reason: format!("invalid gRPC metadata value for `{name}`: {error}"),
            })?;
        header_map.insert(header_name, header_value);
    }
    let metadata = MetadataMap::from_headers(header_map);

    let build_exporters = || -> Result<(SpanExporter, LogExporter, MetricExporter), SKError> {
        let span_exporter = SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint.clone())
            .with_timeout(timeout)
            .with_metadata(metadata.clone())
            .build()
            .map_err(|error| SKError::UnsupportedOperation {
                operation: "telemetry_init",
                reason: format!("failed to build OTLP gRPC span exporter: {error}"),
            })?;
        let log_exporter = LogExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint.clone())
            .with_timeout(timeout)
            .with_metadata(metadata.clone())
            .build()
            .map_err(|error| SKError::UnsupportedOperation {
                operation: "telemetry_init",
                reason: format!("failed to build OTLP gRPC log exporter: {error}"),
            })?;
        let metric_exporter = MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_timeout(timeout)
            .with_metadata(metadata)
            .build()
            .map_err(|error| SKError::UnsupportedOperation {
                operation: "telemetry_init",
                reason: format!("failed to build OTLP gRPC metric exporter: {error}"),
            })?;
        Ok((span_exporter, log_exporter, metric_exporter))
    };

    let (span_exporter, log_exporter, metric_exporter) =
        if tokio::runtime::Handle::try_current().is_ok() {
            build_exporters()?
        } else {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| SKError::UnsupportedOperation {
                    operation: "telemetry_init",
                    reason: format!("failed to initialize gRPC runtime: {error}"),
                })?;
            let _guard = runtime.enter();
            build_exporters()?
        };

    Ok(SKTelemetryPipeline::from_exporters(
        configuration,
        span_exporter,
        log_exporter,
        metric_exporter,
    ))
}

#[cfg(not(feature = "otlp-grpc"))]
fn build_grpc_pipeline(
    _configuration: SKTelemetryConfiguration,
) -> Result<SKTelemetryPipeline, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "telemetry_init",
        reason: "transport `grpc` requires enabling feature `otlp-grpc` or `telemetry`".to_string(),
    })
}

#[cfg(feature = "otlp-http")]
fn build_http_pipeline(
    configuration: SKTelemetryConfiguration,
) -> Result<SKTelemetryPipeline, SKError> {
    use std::collections::HashMap;

    use opentelemetry_otlp::{
        LogExporter, MetricExporter, Protocol, SpanExporter, WithExportConfig, WithHttpConfig,
    };

    let base = configuration.endpoint().trim_end_matches('/').to_string();
    let timeout = configuration.timeout();
    let headers: HashMap<String, String> = configuration
        .headers()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();

    let span_exporter = SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(format!("{base}/v1/traces"))
        .with_timeout(timeout)
        .with_headers(headers.clone())
        .build()
        .map_err(|error| SKError::UnsupportedOperation {
            operation: "telemetry_init",
            reason: format!("failed to build OTLP HTTP span exporter: {error}"),
        })?;
    let log_exporter = LogExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(format!("{base}/v1/logs"))
        .with_timeout(timeout)
        .with_headers(headers.clone())
        .build()
        .map_err(|error| SKError::UnsupportedOperation {
            operation: "telemetry_init",
            reason: format!("failed to build OTLP HTTP log exporter: {error}"),
        })?;
    let metric_exporter = MetricExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(format!("{base}/v1/metrics"))
        .with_timeout(timeout)
        .with_headers(headers)
        .build()
        .map_err(|error| SKError::UnsupportedOperation {
            operation: "telemetry_init",
            reason: format!("failed to build OTLP HTTP metric exporter: {error}"),
        })?;

    Ok(SKTelemetryPipeline::from_exporters(
        configuration,
        span_exporter,
        log_exporter,
        metric_exporter,
    ))
}

#[cfg(not(feature = "otlp-http"))]
fn build_http_pipeline(
    _configuration: SKTelemetryConfiguration,
) -> Result<SKTelemetryPipeline, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "telemetry_init",
        reason: "transport `http` requires enabling feature `otlp-http` or `telemetry`".to_string(),
    })
}
