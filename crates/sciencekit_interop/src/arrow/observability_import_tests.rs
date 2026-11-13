//! TDD suite verifying that Arrow batch conversions emit structured
//! `sciencekit.interop.conversion` tracing spans (spec `dataset-interop-core`).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use tracing::Subscriber;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use super::import::{sk_tabular_batch_from_column_chunks, sk_tabular_batch_from_record_batch};
use crate::missing::SKMissingValuePolicy;

type RecordedSpans = Arc<Mutex<Vec<(String, HashMap<String, String>)>>>;

struct FieldVisitor {
    fields: HashMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

#[derive(Clone, Default)]
struct SpanRecorder {
    spans: RecordedSpans,
    index_by_id: Arc<Mutex<HashMap<Id, usize>>>,
}

impl<S> Layer<S> for SpanRecorder
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, _metadata: &'static tracing::Metadata<'_>) -> Interest {
        Interest::always()
    }

    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, _context: Context<'_, S>) {
        let mut visitor = FieldVisitor {
            fields: HashMap::new(),
        };
        attributes.values().record(&mut visitor);
        let mut spans = self.spans.lock().expect("uncontended lock");
        let position = spans.len();
        spans.push((attributes.metadata().name().to_string(), visitor.fields));
        self.index_by_id
            .lock()
            .expect("uncontended lock")
            .insert(id.clone(), position);
    }

    fn on_record(&self, span: &Id, values: &Record<'_>, _context: Context<'_, S>) {
        let mut visitor = FieldVisitor {
            fields: HashMap::new(),
        };
        values.record(&mut visitor);
        if let Some(&position) = self.index_by_id.lock().expect("uncontended lock").get(span) {
            if let Some((_, existing)) = self
                .spans
                .lock()
                .expect("uncontended lock")
                .get_mut(position)
            {
                existing.extend(visitor.fields);
            }
        }
    }
}

#[test]
fn sk_arrow_batch_and_chunk_conversions_record_conversion_spans() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
        ],
    )
    .expect("a batch");

    let recorder = SpanRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    tracing::subscriber::with_default(subscriber, || {
        let _ = sk_tabular_batch_from_record_batch(&batch, SKMissingValuePolicy::Reject, 0, true)
            .expect("record-batch conversion");
        let chunks = vec![
            vec![Arc::new(Float64Array::from(vec![4.0, 5.0])) as ArrayRef],
            vec![Arc::new(StringArray::from(vec!["d", "e"])) as ArrayRef],
        ];
        let _ = sk_tabular_batch_from_column_chunks(
            schema.as_ref(),
            &chunks,
            SKMissingValuePolicy::NumericNaN,
            1,
            true,
        )
        .expect("chunked conversion");
    });

    let spans = recorder.spans.lock().expect("uncontended lock");
    assert_eq!(spans.len(), 2, "each conversion emits exactly one span");
    assert_eq!(spans[0].0, "sciencekit.interop.conversion");
    assert_eq!(
        spans[0].1.get("source_kind").map(String::as_str),
        Some("arrow")
    );
    assert_eq!(spans[0].1.get("rows").map(String::as_str), Some("3"));
    assert_eq!(spans[0].1.get("columns").map(String::as_str), Some("2"));
    assert_eq!(spans[0].1.get("policy").map(String::as_str), Some("reject"));
    assert!(spans[0].1.contains_key("duration_ms"));

    assert_eq!(spans[1].0, "sciencekit.interop.conversion");
    assert_eq!(
        spans[1].1.get("source_kind").map(String::as_str),
        Some("arrow")
    );
    assert_eq!(spans[1].1.get("rows").map(String::as_str), Some("2"));
    assert_eq!(
        spans[1].1.get("policy").map(String::as_str),
        Some("numeric_nan")
    );
}
