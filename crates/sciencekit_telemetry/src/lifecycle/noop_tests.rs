//! Companion tests proving the silent no-op default without initialization
//! or OTLP transport features (spec `telemetry-export`).

use std::io::ErrorKind;
use std::net::TcpListener;

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::SKTelemetry;
use super::global_state::{GLOBAL_TELEMETRY_TEST_LOCK, sk_reset_global_telemetry_for_tests};
use crate::configuration::{SKTelemetryConfiguration, SKTelemetryTransport};

#[test]
fn uninitialized_default_creates_no_exporter_and_produces_no_network_traffic() {
    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let collector_probe =
        TcpListener::bind("127.0.0.1:0").expect("ephemeral loopback listener binds");
    collector_probe
        .set_nonblocking(true)
        .expect("nonblocking mode succeeds");
    let local_port = collector_probe.local_addr().expect("local addr").port();

    let _unused_configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-noop-service")
        .endpoint(format!("http://127.0.0.1:{local_port}"))
        .transport(SKTelemetryTransport::Http)
        .build()
        .expect("configuration builds without starting any exporter");

    assert!(!SKTelemetry::is_initialized());
    assert_eq!(SKTelemetry::active_exporter_count(), 0);

    let matrix = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let scaler = SKStandardScaler::build().build().expect("scaler builds");
    let fitted = scaler.fit(&matrix).expect("fit succeeds silently");
    let transformed = fitted
        .transform(&matrix)
        .expect("transform succeeds silently");
    assert_eq!(transformed.dim(), (4, 2));
    tracing::info!(rows = 4_i64, "silent operation completed");

    assert!(!SKTelemetry::is_initialized());
    assert_eq!(SKTelemetry::active_exporter_count(), 0);
    let accept_attempt = collector_probe.accept();
    assert!(
        matches!(accept_attempt, Err(ref error) if error.kind() == ErrorKind::WouldBlock),
        "no OTLP connection may be attempted when telemetry is uninitialized, got {accept_attempt:?}"
    );
}

#[cfg(not(any(feature = "otlp-grpc", feature = "otlp-http")))]
#[test]
fn default_feature_set_compiles_out_otlp_network_exporters() {
    use sciencekit_common::SKError;

    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-default-features")
        .build()
        .expect("configuration builds");
    let result = SKTelemetry::init(configuration);
    assert!(matches!(
        result,
        Err(SKError::UnsupportedOperation {
            operation: "telemetry_init",
            ..
        })
    ));
    assert!(!SKTelemetry::is_initialized());
    assert_eq!(SKTelemetry::active_exporter_count(), 0);
}
