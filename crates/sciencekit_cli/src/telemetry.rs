//! Optional OpenTelemetry initialization and flush-on-exit.

use crate::error::SKCliError;

/// The environment variable that turns telemetry on.
pub const SK_OTLP_ENDPOINT_VARIABLE: &str = "SCIENCEKIT_OTLP_ENDPOINT";
/// The environment variable selecting the OTLP transport (`grpc` or `http`).
#[allow(dead_code)]
pub const SK_OTLP_TRANSPORT_VARIABLE: &str = "SCIENCEKIT_OTLP_TRANSPORT";

/// A guard that shuts the telemetry pipeline down and flushes pending signals.
pub struct SKTelemetryGuard {
    #[cfg(feature = "telemetry")]
    handle: Option<sciencekit::telemetry::SKTelemetry>,
}

impl SKTelemetryGuard {
    fn disabled() -> Self {
        Self {
            #[cfg(feature = "telemetry")]
            handle: None,
        }
    }
}

impl Drop for SKTelemetryGuard {
    fn drop(&mut self) {
        #[cfg(feature = "telemetry")]
        if let Some(handle) = self.handle.take() {
            let _ = handle.shutdown();
        }
    }
}

/// Initialize telemetry when `SCIENCEKIT_OTLP_ENDPOINT` is set.
///
/// # Errors
/// Returns an unsupported-feature error when the endpoint is set but the binary
/// was built without the `telemetry` feature, and a runtime error when the
/// configuration or initialization is rejected.
pub fn sk_initialize_telemetry() -> Result<SKTelemetryGuard, SKCliError> {
    let endpoint = match std::env::var(SK_OTLP_ENDPOINT_VARIABLE) {
        Ok(value) if !value.is_empty() => value,
        _ => return Ok(SKTelemetryGuard::disabled()),
    };

    #[cfg(not(feature = "telemetry"))]
    {
        let _ = endpoint;
        Err(SKCliError::unsupported(
            "telemetry",
            format!("`{SK_OTLP_ENDPOINT_VARIABLE}`"),
        ))
    }

    #[cfg(feature = "telemetry")]
    {
        use std::time::Duration;

        use sciencekit::SKBuilder;
        use sciencekit::telemetry::{
            SKTelemetry, SKTelemetryConfiguration, SKTelemetrySampler, SKTelemetryTransport,
        };
        let transport = match std::env::var(SK_OTLP_TRANSPORT_VARIABLE)
            .unwrap_or_else(|_| "grpc".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "grpc" => SKTelemetryTransport::Grpc,
            "http" => SKTelemetryTransport::Http,
            other => {
                return Err(SKCliError::validation(
                    format!("environment.{SK_OTLP_TRANSPORT_VARIABLE}"),
                    format!("`{other}` is not a transport; use `grpc` or `http`"),
                ));
            }
        };
        let configuration = SKTelemetryConfiguration::builder()
            .service_name("sciencekit-cli")
            .service_version(env!("CARGO_PKG_VERSION"))
            .endpoint(endpoint)
            .transport(transport)
            .timeout(Duration::from_secs(5))
            .sampler(SKTelemetrySampler::ParentBased(1.0))
            .build()
            .map_err(|error| {
                SKCliError::validation("environment.SCIENCEKIT_OTLP_ENDPOINT", error.to_string())
            })?;
        let handle = SKTelemetry::init(configuration)
            .map_err(|error| SKCliError::runtime(format!("telemetry initialization: {error}")))?;
        Ok(SKTelemetryGuard {
            handle: Some(handle),
        })
    }
}
