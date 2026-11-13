//! OpenTelemetry metric pipeline and operation instruments (spec `telemetry-export`).

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram, MeterProvider as _};
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};

use crate::configuration::SKTelemetryConfiguration;

/// Histogram recording operation elapsed duration in milliseconds (`ms`).
pub const SK_METRIC_OPERATION_DURATION: &str = "sciencekit.operation.duration";
/// Counter recording total completed library operations.
pub const SK_METRIC_OPERATION_COUNT: &str = "sciencekit.operation.count";
/// Histogram recording input row counts observed by library operations.
pub const SK_METRIC_OPERATION_ROWS: &str = "sciencekit.operation.rows";
/// Histogram recording input column counts observed by library operations.
pub const SK_METRIC_OPERATION_COLUMNS: &str = "sciencekit.operation.columns";
/// Counter recording library operations that terminated with an error.
pub const SK_METRIC_OPERATION_ERRORS: &str = "sciencekit.operation.errors";

pub(crate) fn sk_canonical_operation_label(raw: &str) -> &'static str {
    match raw {
        "fit" => "fit",
        "partial_fit" => "partial_fit",
        "fit_transform" => "fit_transform",
        "transform" => "transform",
        "predict" => "predict",
        "fit_predict" => "fit_predict",
        "score" => "score",
        _ => "unknown",
    }
}

pub(crate) fn sk_canonical_backend_label(raw: &str) -> &'static str {
    raw.parse::<sciencekit_common::SKBackendKind>()
        .map_or("unknown", sciencekit_common::SKBackendKind::name)
}

pub(crate) fn sk_canonical_mode_label(raw: &str) -> &'static str {
    match raw {
        "Automatic" => "Automatic",
        "InProcessSynchronous" => "InProcessSynchronous",
        "InProcessAsynchronous" => "InProcessAsynchronous",
        "OutOfCoreStreaming" => "OutOfCoreStreaming",
        "OutOfCoreMemoryMapped" => "OutOfCoreMemoryMapped",
        _ => "unknown",
    }
}

#[derive(Clone)]
pub(crate) struct SKOperationInstruments {
    duration_milliseconds: Histogram<f64>,
    operation_count: Counter<u64>,
    rows_count: Histogram<u64>,
    columns_count: Histogram<u64>,
    error_count: Counter<u64>,
}

impl SKOperationInstruments {
    pub(crate) fn new(meter_provider: &SdkMeterProvider) -> Self {
        let meter = meter_provider.meter("sciencekit");
        let duration_milliseconds = meter
            .f64_histogram(SK_METRIC_OPERATION_DURATION)
            .with_unit("ms")
            .with_description("Elapsed duration of sciencekit operations in milliseconds.")
            .build();
        let operation_count = meter
            .u64_counter(SK_METRIC_OPERATION_COUNT)
            .with_unit("{operation}")
            .with_description("Total number of completed sciencekit operations.")
            .build();
        let rows_count = meter
            .u64_histogram(SK_METRIC_OPERATION_ROWS)
            .with_unit("{row}")
            .with_description("Input row counts processed by sciencekit operations.")
            .build();
        let columns_count = meter
            .u64_histogram(SK_METRIC_OPERATION_COLUMNS)
            .with_unit("{column}")
            .with_description("Input column counts processed by sciencekit operations.")
            .build();
        let error_count = meter
            .u64_counter(SK_METRIC_OPERATION_ERRORS)
            .with_unit("{error}")
            .with_description("Total number of failed sciencekit operations.")
            .build();
        Self {
            duration_milliseconds,
            operation_count,
            rows_count,
            columns_count,
            error_count,
        }
    }

    pub(crate) fn record_completed_operation(
        &self,
        labels: &[KeyValue],
        duration_milliseconds: f64,
        rows: Option<u64>,
        columns: Option<u64>,
        has_error: bool,
    ) {
        self.duration_milliseconds
            .record(duration_milliseconds, labels);
        self.operation_count.add(1, labels);
        if let Some(row_count) = rows {
            self.rows_count.record(row_count, labels);
        }
        if let Some(column_count) = columns {
            self.columns_count.record(column_count, labels);
        }
        if has_error {
            self.error_count.add(1, labels);
        }
    }
}

/// Build an [`SdkMeterProvider`] configured with periodic export and resource metadata.
pub fn sk_build_meter_provider<E>(
    configuration: &SKTelemetryConfiguration,
    exporter: E,
) -> SdkMeterProvider
where
    E: PushMetricExporter + 'static,
{
    let reader = PeriodicReader::builder(exporter)
        .with_interval(configuration.batch().periodic_metric_interval())
        .build();
    SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(configuration.resource().to_sdk_resource())
        .build()
}
