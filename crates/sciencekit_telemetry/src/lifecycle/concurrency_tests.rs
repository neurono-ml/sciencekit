//! Companion concurrency tests for initialization races, parallel operations,
//! concurrent flushing and clean shutdown (spec `telemetry-export`).

use std::sync::{Arc, Barrier};
use std::thread;

use ndarray::{Array1, Array2};
use opentelemetry_sdk::logs::InMemoryLogExporter;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::trace::InMemorySpanExporter;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFeatureTransformer, SKModelMetadata,
    SKModelState, SKTensor, SKTensorDataType, SKUnsupervisedFit,
};
use sciencekit_metrics::regression::sk_mean_squared_error;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::SKTelemetry;
use super::global_state::{GLOBAL_TELEMETRY_TEST_LOCK, sk_reset_global_telemetry_for_tests};
use crate::configuration::SKTelemetryConfiguration;
use crate::pipeline::SK_METRIC_OPERATION_COUNT;

#[test]
fn concurrent_initialization_race_elects_single_winner_and_rejects_others() {
    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let thread_count = 8_usize;
    let barrier = Arc::new(Barrier::new(thread_count));
    let handles: Vec<_> = (0..thread_count)
        .map(|worker_index| {
            let barrier_clone = Arc::clone(&barrier);
            thread::spawn(move || {
                let configuration = SKTelemetryConfiguration::builder()
                    .service_name(format!("sciencekit-race-{worker_index}"))
                    .build()
                    .expect("configuration builds");
                barrier_clone.wait();
                SKTelemetry::init_with_exporters(
                    configuration,
                    InMemorySpanExporter::default(),
                    InMemoryLogExporter::default(),
                    InMemoryMetricExporter::default(),
                )
            })
        })
        .collect();

    let mut winners = Vec::new();
    let mut rejected = 0_usize;
    for handle in handles {
        match handle.join().expect("worker should not panic") {
            Ok(telemetry) => winners.push(telemetry),
            Err(SKError::UnsupportedOperation {
                operation: "telemetry_init",
                ..
            }) => {
                rejected += 1;
            }
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    assert_eq!(winners.len(), 1);
    assert_eq!(rejected, thread_count - 1);
    let winner = winners.pop().expect("single winner");
    assert!(!winner.is_shut_down());
    winner.shutdown().expect("winner shutdown succeeds");
}

#[test]
fn concurrent_small_and_large_operations_export_consistent_spans_logs_and_metrics() {
    let _guard = GLOBAL_TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sk_reset_global_telemetry_for_tests();

    let span_exporter = InMemorySpanExporter::default();
    let log_exporter = InMemoryLogExporter::default();
    let metric_exporter = InMemoryMetricExporter::default();
    let configuration = SKTelemetryConfiguration::builder()
        .service_name("sciencekit-concurrency-acceptance")
        .service_version("0.7.0")
        .deployment_environment("testing")
        .build()
        .expect("configuration builds");

    let telemetry = Arc::new(
        SKTelemetry::init_with_exporters(
            configuration,
            span_exporter.clone(),
            log_exporter.clone(),
            metric_exporter.clone(),
        )
        .expect("telemetry init succeeds"),
    );

    let worker_count = 6_usize;
    let barrier = Arc::new(Barrier::new(worker_count));
    let handles: Vec<_> = (0..worker_count)
        .map(|worker_index| {
            let barrier_clone = Arc::clone(&barrier);
            let telemetry_clone = Arc::clone(&telemetry);
            thread::spawn(move || {
                let rows = if worker_index % 2 == 0 {
                    8_usize
                } else {
                    1024_usize
                };
                let columns = 4_usize;
                let values: Vec<f64> = (0..rows * columns)
                    .map(|index| (index as f64) * 0.25 + (worker_index as f64))
                    .collect();
                let features =
                    Array2::from_shape_vec((rows, columns), values).expect("matrix shape valid");

                barrier_clone.wait();
                let scaler = SKStandardScaler::build()
                    .backend(SKBackendKind::Faer)
                    .execution_mode(SKExecutionMode::Automatic)
                    .build()
                    .expect("scaler builds");
                let fitted = scaler.fit(&features).expect("fit succeeds");
                let transformed = fitted.transform(&features).expect("transform succeeds");

                let metadata =
                    SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F64, columns)
                        .expect("metadata valid");
                let mean_tensor = SKTensor::floating(
                    "standard_scaler.mean",
                    vec![columns],
                    fitted.mean().to_vec(),
                )
                .expect("tensor valid");
                let exported_state =
                    SKModelState::<f64>::new(metadata, vec![mean_tensor]).expect("state valid");
                assert_eq!(exported_state.tensors().len(), 1);

                let zero_reference = Array1::<f64>::zeros(rows * columns);
                let flattened = Array1::from_vec(transformed.iter().copied().collect());
                let mse = sk_mean_squared_error(zero_reference.view(), flattened.view())
                    .expect("mse succeeds");
                assert!(mse.is_finite() && mse >= 0.0);

                tracing::info!(
                    worker = worker_index as i64,
                    rows = rows as i64,
                    mse = mse,
                    "concurrent worker finished"
                );
                telemetry_clone
                    .force_flush()
                    .expect("concurrent force_flush succeeds");
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("concurrent worker must not panic");
    }

    let telemetry = Arc::try_unwrap(telemetry)
        .unwrap_or_else(|_| panic!("all worker Arc clones must be dropped"));
    telemetry.force_flush().expect("final flush succeeds");

    let spans = span_exporter.get_finished_spans().expect("spans readable");
    let logs = log_exporter.get_emitted_logs().expect("logs readable");
    let metrics = metric_exporter
        .get_finished_metrics()
        .expect("metrics readable");

    assert_eq!(spans.len(), worker_count * 2);
    assert_eq!(logs.len(), worker_count);

    let latest_metrics = metrics
        .last()
        .expect("at least one metric snapshot exported");
    let operation_count_metric = latest_metrics
        .scope_metrics()
        .flat_map(|scope| scope.metrics())
        .find(|metric| metric.name() == SK_METRIC_OPERATION_COUNT)
        .expect("operation count metric exported");
    let AggregatedMetrics::U64(MetricData::Sum(sum)) = operation_count_metric.data() else {
        panic!("expected u64 sum for operation count");
    };
    let recorded_total: u64 = sum.data_points().map(|point| point.value()).sum();
    assert_eq!(recorded_total, (worker_count * 2) as u64);

    telemetry.shutdown().expect("clean shutdown succeeds");
}
