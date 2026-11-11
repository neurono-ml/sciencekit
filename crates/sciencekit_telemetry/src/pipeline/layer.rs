//! `tracing` subscriber layer bridging operation spans to OpenTelemetry metrics
//! (spec `telemetry-export`).

use std::fmt;
use std::time::Instant;

use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use tracing::Event;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

use super::metrics::SKOperationInstruments;

#[derive(Default)]
struct OperationSpanState {
    started_at: Option<Instant>,
    operation: Option<String>,
    backend: Option<String>,
    mode: Option<String>,
    rows: Option<u64>,
    columns: Option<u64>,
    has_error: bool,
}

impl OperationSpanState {
    fn new() -> Self {
        Self {
            started_at: Some(Instant::now()),
            ..Self::default()
        }
    }
}

struct OperationFieldVisitor<'a> {
    state: &'a mut OperationSpanState,
}

impl OperationFieldVisitor<'_> {
    fn assign_text(&mut self, name: &str, value: String) {
        let trimmed = value
            .strip_prefix('"')
            .and_then(|inner| inner.strip_suffix('"'))
            .map(ToOwned::to_owned)
            .unwrap_or(value);
        match name {
            "operation" => self.state.operation = Some(trimmed),
            "backend" => self.state.backend = Some(trimmed),
            "mode" => self.state.mode = Some(trimmed),
            "error" if !trimmed.is_empty() => self.state.has_error = true,
            "rows" => {
                if let Ok(parsed) = trimmed.parse::<u64>() {
                    self.state.rows = Some(parsed);
                }
            }
            "columns" => {
                if let Ok(parsed) = trimmed.parse::<u64>() {
                    self.state.columns = Some(parsed);
                }
            }
            _ => {}
        }
    }
}

impl Visit for OperationFieldVisitor<'_> {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "rows" => self.state.rows = Some(value),
            "columns" => self.state.columns = Some(value),
            _ => {}
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if let Ok(unsigned) = u64::try_from(value) {
            self.record_u64(field, unsigned);
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.assign_text(field.name(), value.to_string());
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        if field.name() == "error" {
            let _ = value;
            self.state.has_error = true;
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.assign_text(field.name(), format!("{value:?}"));
    }
}

/// Subscriber layer observing `sciencekit.operation` spans and recording bounded
/// OpenTelemetry duration, count, shape and error metrics on span close.
#[derive(Clone)]
pub struct SKTelemetryMetricsLayer {
    instruments: SKOperationInstruments,
}

impl SKTelemetryMetricsLayer {
    /// Construct a metrics bridge layer backed by `meter_provider`.
    pub fn new(meter_provider: &SdkMeterProvider) -> Self {
        Self {
            instruments: SKOperationInstruments::new(meter_provider),
        }
    }
}

impl<S> Layer<S> for SKTelemetryMetricsLayer
where
    S: tracing::Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        if attributes.metadata().name() != "sciencekit.operation" {
            return;
        }
        if let Some(span_reference) = context.span(id) {
            let mut state = OperationSpanState::new();
            attributes.record(&mut OperationFieldVisitor { state: &mut state });
            span_reference.extensions_mut().insert(state);
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, context: Context<'_, S>) {
        if let Some(span_reference) = context.span(id) {
            let mut extensions = span_reference.extensions_mut();
            if let Some(state) = extensions.get_mut::<OperationSpanState>() {
                values.record(&mut OperationFieldVisitor { state });
            }
        }
    }

    fn on_event(&self, event: &Event<'_>, context: Context<'_, S>) {
        if *event.metadata().level() == tracing::Level::ERROR {
            if let Some(span_reference) = context.event_span(event) {
                let mut extensions = span_reference.extensions_mut();
                if let Some(state) = extensions.get_mut::<OperationSpanState>() {
                    state.has_error = true;
                }
            }
        }
    }

    fn on_close(&self, id: Id, context: Context<'_, S>) {
        let Some(span_reference) = context.span(&id) else {
            return;
        };
        let Some(state) = span_reference
            .extensions_mut()
            .remove::<OperationSpanState>()
        else {
            return;
        };
        let duration_milliseconds = state
            .started_at
            .map(|started| started.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let operation = state.operation.unwrap_or_else(|| "unknown".to_string());
        let backend = state.backend.unwrap_or_else(|| "unknown".to_string());
        let mode = state.mode.unwrap_or_else(|| "unknown".to_string());
        let labels = [
            KeyValue::new("operation", operation),
            KeyValue::new("backend", backend),
            KeyValue::new("mode", mode),
        ];
        self.instruments.record_completed_operation(
            &labels,
            duration_milliseconds,
            state.rows,
            state.columns,
            state.has_error,
        );
    }
}

/// Build a boxed `tracing` layer exporting operation metrics to `meter_provider`.
pub fn sk_build_metrics_layer<S>(
    meter_provider: &SdkMeterProvider,
) -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: tracing::Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    Box::new(SKTelemetryMetricsLayer::new(meter_provider))
}
