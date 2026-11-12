//! Companion test verifying bounded `force_flush` timeout behavior
//! (spec `telemetry-export`).

use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ndarray::array;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::logs::InMemoryLogExporter;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use sciencekit_common::{SKBuilder, SKError, SKUnsupervisedFit};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::SKTelemetry;
use super::global_state::{GLOBAL_TELEMETRY_TEST_LOCK, sk_reset_global_telemetry_for_tests};
use crate::configuration::SKTelemetryConfiguration;

#[derive(Clone, Debug)]
struct StallingSpanExporter {
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl StallingSpanExporter {
    fn new() -> (Self, Arc<(Mutex<bool>, Condvar)>) {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        (Self { gate: gate.clone() }, gate)
    }
}

impl SpanExporter for StallingSpanExporter {
    fn export(&self, _batch: Vec<SpanData>) -> impl Future<Output = OTelSdkResult> + Send {
        let gate = Arc::clone(&self.gate);
        async move {
            let (lock, condvar) = &*gate;
            let released = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let _ = condvar
                .wait_timeout_while(released, Duration::from_secs(2), |&mut unblocked| {
                    !unblocked
                })
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Err(OTelSdkError::InternalFailure(
                "stalled exporter".to_string(),
            ))
        }
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn set_resource(&mut self, _resource: &Resource) {}
}

#[test]
fn force_flush_times_out_promptly_when_exporter_exceeds_configured_flush_timeout() {
    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let (stalling_exporter, gate) = StallingSpanExporter::new();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-flush-timeout-test")
        .flush_timeout(Duration::from_millis(30))
        .build()
        .expect("configuration builds");

    let telemetry = SKTelemetry::init_with_exporters(
        configuration,
        stalling_exporter,
        InMemoryLogExporter::default(),
        InMemoryMetricExporter::default(),
    )
    .expect("init succeeds");

    let matrix = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let scaler = SKStandardScaler::build().build().expect("scaler builds");
    let _ = scaler.fit(&matrix).expect("fit succeeds");

    let started = Instant::now();
    let flush_result = telemetry.force_flush();
    let elapsed = started.elapsed();

    let (lock, condvar) = &*gate;
    *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
    condvar.notify_all();

    assert!(
        matches!(
            flush_result,
            Err(SKError::UnsupportedOperation {
                operation: "telemetry_force_flush",
                ..
            })
        ),
        "expected structured flush error, got {flush_result:?}"
    );
    assert!(
        elapsed < Duration::from_millis(500),
        "force_flush must honor configured 30ms flush_timeout, took {elapsed:?}"
    );

    let _ = telemetry.shutdown();
}
