//! A minimal span recorder shared by the observability tests.
//!
//! A subscriber layer captures every span's name and recorded fields so tests
//! assert exactly what a consumer's subscriber observes.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Captured spans, keyed by span id: `(name, recorded fields)`.
type RecordedSpans = Arc<Mutex<HashMap<Id, (String, Vec<(String, String)>)>>>;

/// Collects every field value of a span/record into `(name, debug_string)`.
struct FieldCollector {
    fields: Vec<(&'static str, String)>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
    }
}

/// A minimal layer recording each span's name and its recorded fields.
#[derive(Clone, Default)]
pub(crate) struct RecordingLayer {
    spans: RecordedSpans,
}

impl RecordingLayer {
    /// Every recorded span: `(name, fields)`.
    pub(crate) fn recorded(&self) -> Vec<(String, Vec<(String, String)>)> {
        self.spans.lock().unwrap().values().cloned().collect()
    }

    /// The recorded fields of the single captured span, as a lookup map.
    pub(crate) fn single_span_fields(&self) -> HashMap<String, String> {
        let spans = self.recorded();
        assert_eq!(spans.len(), 1, "exactly one span must be recorded");
        spans[0].1.iter().cloned().collect()
    }
}

impl<S> Layer<S> for RecordingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, _context: Context<'_, S>) {
        let mut collector = FieldCollector { fields: Vec::new() };
        attributes.values().record(&mut collector);
        let name = attributes.metadata().name().to_string();
        let fields = collector
            .fields
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        self.spans
            .lock()
            .unwrap()
            .insert(id.clone(), (name, fields));
    }

    fn on_record(&self, span: &Id, values: &Record<'_>, _context: Context<'_, S>) {
        let mut collector = FieldCollector { fields: Vec::new() };
        values.record(&mut collector);
        if let Some((_, fields)) = self.spans.lock().unwrap().get_mut(span) {
            fields.extend(
                collector
                    .fields
                    .into_iter()
                    .map(|(name, value)| (name.to_string(), value)),
            );
        }
    }
}
