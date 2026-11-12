//! Shared process-wide tracing recorder for PCA span and warning tests.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

type RecordedSpans = Arc<Mutex<Vec<(String, Vec<(String, String)>)>>>;
type RecordedEvents = Arc<Mutex<Vec<(tracing::Level, Vec<(String, String)>)>>>;

static TRACING_TEST_LOCK: Mutex<()> = Mutex::new(());
static TRACING_RECORDER: OnceLock<PcaTracingLayer> = OnceLock::new();

struct FieldVisitor {
    fields: Vec<(&'static str, String)>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
    }
}

#[derive(Clone, Default)]
pub(super) struct PcaTracingLayer {
    spans: RecordedSpans,
    events: RecordedEvents,
}

impl<S> Layer<S> for PcaTracingLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        let mut visitor = FieldVisitor { fields: Vec::new() };
        attributes.values().record(&mut visitor);
        let fields = visitor
            .fields
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        self.spans
            .lock()
            .unwrap()
            .push((attributes.metadata().name().to_string(), fields));
    }

    fn on_record(&self, _span: &Id, _values: &Record<'_>, _context: Context<'_, S>) {}

    fn on_event(&self, event: &tracing::Event<'_>, _context: Context<'_, S>) {
        let mut visitor = FieldVisitor { fields: Vec::new() };
        event.record(&mut visitor);
        let fields = visitor
            .fields
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        self.events
            .lock()
            .unwrap()
            .push((*event.metadata().level(), fields));
    }
}

pub(super) struct PcaCaptureGuard {
    _lock: MutexGuard<'static, ()>,
    layer: &'static PcaTracingLayer,
}

impl PcaCaptureGuard {
    pub(super) fn begin() -> Self {
        let lock = TRACING_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let layer = TRACING_RECORDER.get_or_init(|| {
            let recorder = PcaTracingLayer::default();
            let subscriber = tracing_subscriber::registry().with(recorder.clone());
            tracing::subscriber::set_global_default(subscriber)
                .expect("PCA tracing layer is installed once per test binary");
            tracing::callsite::rebuild_interest_cache();
            recorder
        });
        layer.spans.lock().unwrap().clear();
        layer.events.lock().unwrap().clear();
        Self { _lock: lock, layer }
    }

    pub(super) fn spans(&self) -> Vec<(String, HashMap<String, String>)> {
        self.layer
            .spans
            .lock()
            .unwrap()
            .iter()
            .map(|(name, fields)| {
                let map = fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                (name.clone(), map)
            })
            .collect()
    }

    pub(super) fn warning_events(&self) -> Vec<HashMap<String, String>> {
        self.layer
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|(level, _)| *level == tracing::Level::WARN)
            .map(|(_, fields)| {
                fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .collect()
    }
}
