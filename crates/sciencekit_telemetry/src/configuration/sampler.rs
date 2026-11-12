//! Trace sampling strategies for telemetry export (spec `telemetry-export`).

use opentelemetry_sdk::trace::Sampler;
use sciencekit_common::SKError;
use sciencekit_common::builders::sk_validate_hyperparameter;

/// Sampling policy applied to newly created root and child spans.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SKTelemetrySampler {
    /// Sample every span (`1.0`).
    #[default]
    AlwaysOn,
    /// Drop every span (`0.0`).
    AlwaysOff,
    /// Sample a deterministic fraction of traces in `[0.0, 1.0]` by trace identifier.
    TraceIdRatioBased(f64),
    /// Respect an upstream parent's sampling decision, falling back to a
    /// trace-id ratio in `[0.0, 1.0]` for root spans.
    ParentBased(f64),
}

impl SKTelemetrySampler {
    /// Validate that any ratio parameter is finite and lies in `[0.0, 1.0]`.
    pub fn validate(self) -> Result<(), SKError> {
        match self {
            SKTelemetrySampler::AlwaysOn | SKTelemetrySampler::AlwaysOff => Ok(()),
            SKTelemetrySampler::TraceIdRatioBased(ratio)
            | SKTelemetrySampler::ParentBased(ratio) => sk_validate_hyperparameter(
                "sampler",
                ratio.is_finite() && (0.0..=1.0).contains(&ratio),
                format!("sampling ratio must be finite and in [0.0, 1.0], found {ratio}"),
            ),
        }
    }

    /// Convert into the [`opentelemetry_sdk::trace::Sampler`] implementation.
    pub fn to_sdk_sampler(self) -> Sampler {
        match self {
            SKTelemetrySampler::AlwaysOn => Sampler::AlwaysOn,
            SKTelemetrySampler::AlwaysOff => Sampler::AlwaysOff,
            SKTelemetrySampler::TraceIdRatioBased(ratio) => Sampler::TraceIdRatioBased(ratio),
            SKTelemetrySampler::ParentBased(ratio) => {
                Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(ratio)))
            }
        }
    }
}
