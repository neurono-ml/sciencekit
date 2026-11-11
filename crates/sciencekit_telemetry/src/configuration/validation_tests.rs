//! Companion invalid-value tests for telemetry configuration builders.

use std::time::Duration;

use sciencekit_common::{SKBuilder, SKError};

use super::{SKTelemetryBatchConfiguration, SKTelemetryConfiguration, SKTelemetrySampler};

#[test]
fn configuration_builder_rejects_missing_or_blank_service_name() {
    let missing_error = SKTelemetryConfiguration::builder()
        .build()
        .expect_err("missing service name must be rejected");
    assert!(matches!(
        missing_error,
        SKError::InvalidHyperparameter {
            name: "service_name",
            ..
        }
    ));

    let blank_error = SKTelemetryConfiguration::builder()
        .service_name("   ")
        .build()
        .expect_err("blank service name must be rejected");
    assert!(matches!(
        blank_error,
        SKError::InvalidHyperparameter {
            name: "service_name",
            ..
        }
    ));
}

#[test]
fn configuration_builder_rejects_malformed_endpoints() {
    for invalid_endpoint in [
        "",
        "localhost:4317",
        "ftp://localhost:4317",
        "http://",
        "https:///v1/traces",
        "http://bad host:4317",
        "http://localhost:0",
        "http://localhost:70000",
        "http://localhost:notaport",
    ] {
        let error = SKTelemetryConfiguration::builder()
            .service_name("sciencekit")
            .endpoint(invalid_endpoint)
            .build()
            .expect_err("malformed endpoint must fail validation");
        assert!(
            matches!(
                error,
                SKError::InvalidHyperparameter {
                    name: "endpoint",
                    ..
                }
            ),
            "expected endpoint error for `{invalid_endpoint}`, got {error:?}"
        );
    }
}

#[test]
fn configuration_builder_rejects_invalid_headers_timeouts_samplers_and_batch_bounds() {
    let header_error = SKTelemetryConfiguration::builder()
        .service_name("sciencekit")
        .header("bad header", "value")
        .build()
        .expect_err("header name with whitespace must fail");
    assert!(matches!(
        header_error,
        SKError::InvalidHyperparameter {
            name: "headers",
            ..
        }
    ));

    let crlf_header_error = SKTelemetryConfiguration::builder()
        .service_name("sciencekit")
        .header("x-token", "line1\r\nline2")
        .build()
        .expect_err("header value with CRLF must fail");
    assert!(matches!(
        crlf_header_error,
        SKError::InvalidHyperparameter {
            name: "headers",
            ..
        }
    ));

    let timeout_error = SKTelemetryConfiguration::builder()
        .service_name("sciencekit")
        .timeout(Duration::ZERO)
        .build()
        .expect_err("zero timeout must fail");
    assert!(matches!(
        timeout_error,
        SKError::InvalidHyperparameter {
            name: "timeout",
            ..
        }
    ));

    let sampler_error = SKTelemetryConfiguration::builder()
        .service_name("sciencekit")
        .sampler(SKTelemetrySampler::ParentBased(1.5))
        .build()
        .expect_err("out-of-range sampler must fail");
    assert!(matches!(
        sampler_error,
        SKError::InvalidHyperparameter {
            name: "sampler",
            ..
        }
    ));

    let batch_error = SKTelemetryBatchConfiguration::builder()
        .maximum_queue_size(64)
        .maximum_export_batch_size(128)
        .build()
        .expect_err("export batch exceeding queue size must fail");
    assert!(matches!(
        batch_error,
        SKError::InvalidHyperparameter {
            name: "maximum_export_batch_size",
            ..
        }
    ));
}
