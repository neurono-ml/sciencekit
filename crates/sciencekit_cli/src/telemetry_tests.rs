//! Telemetry lifecycle tests.
//!
//! The in-memory exporter test runs only with the `telemetry` feature and
//! proves the same init/emit/flush/shutdown path the binary uses.

#[cfg(not(feature = "telemetry"))]
#[test]
fn a_configured_endpoint_without_the_feature_is_unsupported() {
    let error = crate::error::SKCliError::unsupported("telemetry", "SCIENCEKIT_OTLP_ENDPOINT");
    assert_eq!(
        error.exit_code(),
        crate::cli::exit_codes::SKExitCode::UnsupportedFeature
    );
}

#[cfg(feature = "telemetry")]
#[test]
fn in_memory_exporters_flush_and_shut_down() {
    use opentelemetry_sdk::logs::InMemoryLogExporter;
    use opentelemetry_sdk::metrics::InMemoryMetricExporter;
    use opentelemetry_sdk::trace::InMemorySpanExporter;
    use sciencekit::SKBuilder;
    use sciencekit::telemetry::{SKTelemetry, SKTelemetryConfiguration};

    let spans = InMemorySpanExporter::default();
    let logs = InMemoryLogExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-cli-test")
        .build()
        .expect("the configuration builds");
    let telemetry = SKTelemetry::init_with_exporters(
        configuration,
        spans.clone(),
        logs.clone(),
        InMemoryMetricExporter::default(),
    )
    .expect("telemetry initializes");
    tracing::info!(target: "sciencekit_cli::tests", "cli telemetry probe");
    telemetry.force_flush().expect("telemetry flushes");
    telemetry.shutdown().expect("telemetry shuts down");
    assert!(!SKTelemetry::is_initialized());
}
