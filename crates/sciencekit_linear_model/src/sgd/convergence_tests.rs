//! Companion tests for the convergence bookkeeping (spec
//! `stochastic-gradient-solver`, decision 7).

use std::fmt;
use std::sync::{Arc, Mutex};

use sciencekit_common::SKError;
use tracing::Subscriber;
use tracing::field::Visit;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use super::convergence::{SKConvergencePolicy, SKConvergenceTracker, sk_finish_epochs};

#[test]
fn a_plateau_converges_before_the_budget() {
    let mut tracker = SKConvergenceTracker::<f64>::new(0.1, 2);
    assert!(!tracker.record(5.0));
    assert!(!tracker.record(4.0));
    // Two consecutive epochs improving by no more than the tolerance.
    assert!(!tracker.record(3.95));
    assert!(tracker.record(3.90));
    assert!(tracker.converged());
    assert_eq!(tracker.epochs_without_improvement(), 2);
}

#[test]
fn strict_exhaustion_raises_not_converged_with_the_iterations() {
    let error = sk_finish_epochs(SKConvergencePolicy::Strict, 10, 10, false, false).unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations } if iterations == 10));
}

#[test]
fn permissive_exhaustion_returns_the_last_iterate() {
    let converged =
        sk_finish_epochs(SKConvergencePolicy::Permissive, 10, 10, false, false).unwrap();
    assert!(!converged);
}

#[test]
fn a_provider_limited_fit_never_raises_even_under_strict() {
    let converged = sk_finish_epochs(SKConvergencePolicy::Strict, 1000, 1, false, true).unwrap();
    assert!(!converged);
}

#[test]
fn a_converged_fit_reports_success_even_under_strict() {
    assert!(sk_finish_epochs(SKConvergencePolicy::Strict, 10, 3, true, false).unwrap());
}

#[test]
fn strict_is_the_default_policy() {
    assert_eq!(SKConvergencePolicy::default(), SKConvergencePolicy::Strict);
}

// A minimal `tracing` layer confined to this file captures every warning event
// so the tests assert exactly what a consumer's subscriber observes.

/// Captured event fields, one entry per event.
type RecordedEvents = Arc<Mutex<Vec<Vec<(String, String)>>>>;

struct EventVisitor {
    fields: Vec<(&'static str, String)>,
}

impl Visit for EventVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields.push((field.name(), format!("{value:?}")));
    }
}

#[derive(Clone, Default)]
struct WarningLayer {
    events: RecordedEvents,
}

impl WarningLayer {
    fn recorded(&self) -> Vec<Vec<(String, String)>> {
        self.events.lock().unwrap().clone()
    }
}

impl<S> Layer<S> for WarningLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _context: Context<'_, S>) {
        let mut visitor = EventVisitor { fields: Vec::new() };
        event.record(&mut visitor);
        self.events.lock().unwrap().push(
            visitor
                .fields
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
        );
    }
}

fn captured_warnings<F: FnOnce()>(record: F) -> Vec<Vec<(String, String)>> {
    let layer = WarningLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());
    tracing::subscriber::with_default(subscriber, record);
    layer.recorded()
}

fn field<'a>(event: &'a [(String, String)], name: &str) -> &'a str {
    event
        .iter()
        .find(|(field, _)| field == name)
        .map(|(_, value)| value.as_str())
        .expect("the field must be recorded")
}

#[test]
fn permissive_exhaustion_emits_a_warning() {
    let events = captured_warnings(|| {
        let converged =
            sk_finish_epochs(SKConvergencePolicy::Permissive, 10, 10, false, false).unwrap();
        assert!(!converged);
    });
    assert_eq!(events.len(), 1, "exactly one warning");
    assert_eq!(field(&events[0], "epochs_taken"), "10");
    assert_eq!(field(&events[0], "requested_budget"), "10");
    assert!(
        field(&events[0], "message").contains("exhausted its budget without converging"),
        "the message must explain the fallback"
    );
}

#[test]
fn a_provider_limited_fit_emits_a_warning() {
    let events = captured_warnings(|| {
        let converged =
            sk_finish_epochs(SKConvergencePolicy::Strict, 1000, 1, false, true).unwrap();
        assert!(!converged);
    });
    assert_eq!(events.len(), 1, "exactly one warning");
    assert_eq!(field(&events[0], "epochs_taken"), "1");
    assert!(
        field(&events[0], "message").contains("supplied fewer passes than the requested budget"),
        "the message must explain the fallback"
    );
}

#[test]
fn a_converged_fit_emits_no_warning() {
    let events = captured_warnings(|| {
        assert!(sk_finish_epochs(SKConvergencePolicy::Permissive, 10, 3, true, false).unwrap());
    });
    assert!(events.is_empty(), "convergence is never warned about");
}
