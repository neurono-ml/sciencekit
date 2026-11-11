//! Flush, shutdown and drop cleanup for [`super::SKTelemetry`]
//! (spec `telemetry-export`).

use std::sync::atomic::Ordering;

use sciencekit_common::SKError;

use super::SKTelemetry;
use super::global_state::sk_detach_global_subscriber;

impl SKTelemetry {
    /// Flush pending traces, logs and metrics to the configured exporters.
    pub fn force_flush(&self) -> Result<(), SKError> {
        if self.is_shut_down.load(Ordering::Acquire) {
            return Err(SKError::UnsupportedOperation {
                operation: "telemetry_force_flush",
                reason: "telemetry pipeline has already been shut down".to_string(),
            });
        }
        self.pipeline.force_flush()
    }

    /// Flush pending signals, detach the global subscriber layer and shut down
    /// all three OpenTelemetry providers.
    pub fn shutdown(self) -> Result<(), SKError> {
        if self.is_shut_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        sk_detach_global_subscriber();
        self.pipeline.shutdown()
    }
}

impl Drop for SKTelemetry {
    fn drop(&mut self) {
        if !self.is_shut_down.swap(true, Ordering::AcqRel) {
            sk_detach_global_subscriber();
            let _ = self.pipeline.shutdown();
        }
    }
}
