//! A test-only span recorder shared by the observability and fallback tests.
//!
//! It captures each span's name and every recorded field, so a test can assert
//! what a consumer's subscriber actually observes. Kept in its own module
//! because both the device-success path and the no-device path need it, in
//! every feature configuration.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// The captured spans keyed by span id: name plus recorded fields.
pub(crate) type CapturedSpans = Arc<Mutex<HashMap<Id, (String, Vec<(String, String)>)>>>;

/// A minimal span recorder: name plus every recorded field.
#[derive(Clone, Default)]
pub(crate) struct RecordingLayer {
    spans: CapturedSpans,
}

impl RecordingLayer {
    /// The captured spans as `(name, fields)` pairs.
    pub(crate) fn spans(&self) -> Vec<(String, HashMap<String, String>)> {
        self.spans
            .lock()
            .unwrap()
            .values()
            .map(|(name, fields)| (name.clone(), fields.iter().cloned().collect()))
            .collect()
    }
}

struct FieldCollector {
    fields: Vec<(&'static str, String)>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
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
        self.spans.lock().unwrap().insert(
            id.clone(),
            (
                name,
                collector
                    .fields
                    .into_iter()
                    .map(|(field, value)| (field.to_string(), value))
                    .collect(),
            ),
        );
    }

    fn on_record(&self, span: &Id, values: &Record<'_>, _context: Context<'_, S>) {
        let mut collector = FieldCollector { fields: Vec::new() };
        values.record(&mut collector);
        if let Some((_, fields)) = self.spans.lock().unwrap().get_mut(span) {
            fields.extend(
                collector
                    .fields
                    .into_iter()
                    .map(|(field, value)| (field.to_string(), value)),
            );
        }
    }
}
