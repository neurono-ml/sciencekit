//! OTLP wire transport selection (spec `telemetry-export`).

use std::fmt;

/// Wire transport protocol used to export telemetry signals to an OTLP collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKTelemetryTransport {
    /// OTLP over gRPC (default port `4317`).
    #[default]
    Grpc,
    /// OTLP over HTTP with protobuf payloads (default port `4318`).
    Http,
}

impl SKTelemetryTransport {
    /// Static protocol identifier (`"grpc"` or `"http"`).
    pub fn name(self) -> &'static str {
        match self {
            SKTelemetryTransport::Grpc => "grpc",
            SKTelemetryTransport::Http => "http",
        }
    }

    /// Standard OTLP collector endpoint for this transport.
    pub fn default_endpoint(self) -> &'static str {
        match self {
            SKTelemetryTransport::Grpc => "http://localhost:4317",
            SKTelemetryTransport::Http => "http://localhost:4318",
        }
    }
}

impl fmt::Display for SKTelemetryTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
