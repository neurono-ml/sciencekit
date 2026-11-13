//! Process-wide delegating subscriber slot and double-initialization guard
//! (spec `telemetry-export`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use sciencekit_common::SKError;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Event, Metadata, Subscriber};

type SharedSubscriber = Arc<dyn Subscriber + Send + Sync>;

fn is_internal_transport_target(target: &str) -> bool {
    target.starts_with("opentelemetry")
        || target.starts_with("tonic")
        || target.starts_with("hyper")
        || target.starts_with("h2")
        || target.starts_with("reqwest")
}

struct SKGlobalTelemetrySubscriber {
    delegate: RwLock<Option<SharedSubscriber>>,
}

impl SKGlobalTelemetrySubscriber {
    fn new() -> Self {
        Self {
            delegate: RwLock::new(None),
        }
    }

    fn current_delegate(&self) -> Option<SharedSubscriber> {
        self.delegate
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn swap(&self, next: Option<SharedSubscriber>) {
        let previous = {
            let mut guard = self
                .delegate
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::replace(&mut *guard, next)
        };
        drop(previous);
    }
}

#[derive(Clone)]
struct SubscriberProxy(Arc<SKGlobalTelemetrySubscriber>);

impl Subscriber for SubscriberProxy {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        if is_internal_transport_target(metadata.target()) {
            return false;
        }
        self.0
            .current_delegate()
            .is_some_and(|subscriber| subscriber.enabled(metadata))
    }

    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if is_internal_transport_target(metadata.target()) {
            return Interest::never();
        }
        Interest::sometimes()
    }

    fn new_span(&self, span: &Attributes<'_>) -> Id {
        self.0
            .current_delegate()
            .map_or_else(|| Id::from_u64(1), |subscriber| subscriber.new_span(span))
    }

    fn record(&self, span: &Id, values: &Record<'_>) {
        if let Some(subscriber) = self.0.current_delegate() {
            subscriber.record(span, values);
        }
    }

    fn record_follows_from(&self, span: &Id, follows: &Id) {
        if let Some(subscriber) = self.0.current_delegate() {
            subscriber.record_follows_from(span, follows);
        }
    }

    fn event(&self, event: &Event<'_>) {
        if is_internal_transport_target(event.metadata().target()) {
            return;
        }
        if let Some(subscriber) = self.0.current_delegate() {
            subscriber.event(event);
        }
    }

    fn enter(&self, span: &Id) {
        if let Some(subscriber) = self.0.current_delegate() {
            subscriber.enter(span);
        }
    }

    fn exit(&self, span: &Id) {
        if let Some(subscriber) = self.0.current_delegate() {
            subscriber.exit(span);
        }
    }

    fn clone_span(&self, id: &Id) -> Id {
        self.0
            .current_delegate()
            .map_or_else(|| id.clone(), |subscriber| subscriber.clone_span(id))
    }

    fn try_close(&self, id: Id) -> bool {
        self.0
            .current_delegate()
            .is_some_and(|subscriber| subscriber.try_close(id))
    }
}

static EVER_INITIALIZED: AtomicBool = AtomicBool::new(false);
static ACTIVE_INITIALIZED: AtomicBool = AtomicBool::new(false);
static ACTIVE_EXPORTER_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static GLOBAL_SUBSCRIBER: OnceLock<Arc<SKGlobalTelemetrySubscriber>> = OnceLock::new();

pub(crate) fn sk_is_active_initialized() -> bool {
    ACTIVE_INITIALIZED.load(Ordering::Acquire)
}

pub(crate) fn sk_active_exporter_count() -> usize {
    ACTIVE_EXPORTER_COUNT.load(Ordering::Acquire)
}

pub(crate) fn sk_claim_initialization_slot() -> Result<(), SKError> {
    EVER_INITIALIZED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map(|_| ())
        .map_err(|_| SKError::UnsupportedOperation {
            operation: "telemetry_init",
            reason: "global telemetry pipeline is already initialized".to_string(),
        })
}

pub(crate) fn sk_release_failed_initialization_slot() {
    EVER_INITIALIZED.store(false, Ordering::Release);
}

pub(crate) fn sk_install_global_subscriber(
    subscriber: Box<dyn Subscriber + Send + Sync + 'static>,
) -> Result<(), SKError> {
    let slot = if let Some(existing) = GLOBAL_SUBSCRIBER.get() {
        existing
    } else {
        let created = Arc::new(SKGlobalTelemetrySubscriber::new());
        tracing::subscriber::set_global_default(SubscriberProxy(created.clone())).map_err(
            |error| SKError::UnsupportedOperation {
                operation: "telemetry_init",
                reason: format!("global tracing subscriber is already installed: {error}"),
            },
        )?;
        let _ = GLOBAL_SUBSCRIBER.set(created);
        GLOBAL_SUBSCRIBER.get().expect("global subscriber set")
    };
    slot.swap(Some(Arc::from(subscriber)));
    tracing::callsite::rebuild_interest_cache();
    ACTIVE_EXPORTER_COUNT.store(3, Ordering::Release);
    ACTIVE_INITIALIZED.store(true, Ordering::Release);
    Ok(())
}

pub(crate) fn sk_detach_global_subscriber() {
    ACTIVE_INITIALIZED.store(false, Ordering::Release);
    ACTIVE_EXPORTER_COUNT.store(0, Ordering::Release);
    if let Some(slot) = GLOBAL_SUBSCRIBER.get() {
        slot.swap(None);
        tracing::callsite::rebuild_interest_cache();
    }
}

#[cfg(test)]
pub(crate) static GLOBAL_TELEMETRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn sk_reset_global_telemetry_for_tests() {
    sk_detach_global_subscriber();
    EVER_INITIALIZED.store(false, Ordering::Release);
}
