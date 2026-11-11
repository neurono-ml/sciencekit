//! Tests for the GPU-aware operation span (spec `gpu-compute-contracts`).

use tracing_subscriber::prelude::*;

use super::{
    SKBackendKind, SKOperationAttributes, SKOperationKind, SKOperationObservation, sk_run_operation,
};
use crate::device::{
    SKDeviceCapabilities, SKDeviceContext, SKDeviceInfo, SKFallbackReason,
    lock_device_inventory_for_test, sk_register_device_inventory,
};
use crate::execution::{
    SKAccessPattern, SKExecutionContext, SKExecutionMode, sk_resolve_execution_plan,
};
use crate::observability::recording_layer::RecordingLayer;

fn attributes(backend: SKBackendKind) -> SKOperationAttributes {
    SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 32,
        columns_count: 8,
        execution_mode: SKExecutionMode::InProcessSynchronous,
        backend,
    }
}

fn context(device: Option<SKDeviceContext>) -> SKExecutionContext {
    SKExecutionContext {
        available_memory_bytes: 1 << 30,
        cpu_cores: 4,
        dataset_size_bytes: 1024,
        dataset_elements: Some(1024),
        scalar_size_bytes: 8,
        grain: 256,
        access_pattern: SKAccessPattern::Sequential,
        batch_size_hint: None,
        batch_byte_size_hint: None,
        device,
    }
}

fn mock_opencl_device() -> SKDeviceInfo {
    SKDeviceInfo::new(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1 << 20,
        SKDeviceCapabilities::new(true, true, false),
    )
}

/// When an OpenCL device is registered, the span records `opencl` and no fallback.
#[test]
fn the_selected_device_kind_is_recorded_in_the_span() {
    let _guard = lock_device_inventory_for_test();
    sk_register_device_inventory(vec![mock_opencl_device()]);

    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let observation = SKOperationObservation::begin(attributes(SKBackendKind::OpenCl));
        observation.in_scope(|| ());
        observation.finish();
    });

    let fields = recording.single_span_fields();
    assert_eq!(fields.get("backend").map(String::as_str), Some("opencl"));
    assert_eq!(fields.get("fallback_reason"), None);
}

/// Without a registered device, a requested GPU kind falls back to `faer` with a reason.
#[test]
fn unregistered_device_kind_falls_back_to_host_and_records_reason() {
    let _guard = lock_device_inventory_for_test();

    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        sk_run_operation(attributes(SKBackendKind::OpenCl), || {
            Ok::<_, crate::SKError>(())
        })
        .unwrap();
    });

    let fields = recording.single_span_fields();
    assert_eq!(fields.get("backend").map(String::as_str), Some("faer"));
    assert_eq!(
        fields.get("fallback_reason").map(String::as_str),
        Some("no-device-available")
    );
}

/// A configured host backend is recorded unchanged and carries no fallback reason.
#[test]
fn host_backend_kind_is_recorded_unchanged_without_fallback() {
    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let observation = SKOperationObservation::begin(attributes(SKBackendKind::MatrixMultiply));
        observation.in_scope(|| ());
        observation.finish();
    });

    let fields = recording.single_span_fields();
    assert_eq!(
        fields.get("backend").map(String::as_str),
        Some("matrixmultiply")
    );
    assert_eq!(fields.get("fallback_reason"), None);
}

/// A fallback reason is recorded on the span as a stable token and never double-recorded.
#[test]
fn the_fallback_reason_is_recorded_once_on_the_span() {
    let _guard = lock_device_inventory_for_test();

    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let observation = SKOperationObservation::begin(attributes(SKBackendKind::OpenCl));
        observation.record_fallback_reason(SKFallbackReason::KernelNotSupported);
        observation.in_scope(|| ());
        observation.finish();
    });

    let recorded = recording.recorded();
    let fallback_entries: Vec<_> = recorded[0]
        .1
        .iter()
        .filter(|(name, _)| name == "fallback_reason")
        .collect();
    assert_eq!(
        fallback_entries.len(),
        1,
        "fallback_reason must not double-record"
    );
    let fields = recording.single_span_fields();
    assert_eq!(fields.get("backend").map(String::as_str), Some("faer"));
    assert_eq!(
        fields.get("fallback_reason").map(String::as_str),
        Some("no-device-available")
    );
}

/// A plan that selects a device records its kind and no fallback reason.
#[test]
fn a_device_plan_records_its_backend_and_no_fallback() {
    let _guard = lock_device_inventory_for_test();
    let inventory = vec![mock_opencl_device()];
    sk_register_device_inventory(inventory.clone());
    let context = context(Some(
        SKDeviceContext::new(SKBackendKind::OpenCl).with_inventory(inventory),
    ));
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap();

    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let observation = SKOperationObservation::begin(attributes(plan.backend));
        if let Some(reason) = plan.fallback_reason {
            observation.record_fallback_reason(reason);
        }
        observation.in_scope(|| ());
        observation.finish();
    });

    let fields = recording.single_span_fields();
    assert_eq!(fields.get("backend").map(String::as_str), Some("opencl"));
    assert_eq!(fields.get("fallback_reason"), None);
}

/// A plan that falls back records the reason it carries.
#[test]
fn a_fallback_plan_records_the_reason_it_carries() {
    let _guard = lock_device_inventory_for_test();
    let context = context(Some(SKDeviceContext::new(SKBackendKind::Cuda)));
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap();
    assert_eq!(
        plan.fallback_reason,
        Some(SKFallbackReason::NoDeviceAvailable)
    );

    let recording = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(recording.clone());
    tracing::subscriber::with_default(subscriber, || {
        let observation = SKOperationObservation::begin(attributes(plan.backend));
        if let Some(reason) = plan.fallback_reason {
            observation.record_fallback_reason(reason);
        }
        observation.in_scope(|| ());
        observation.finish();
    });

    let fields = recording.single_span_fields();
    assert_eq!(fields.get("backend").map(String::as_str), Some("faer"));
    assert_eq!(
        fields.get("fallback_reason").map(String::as_str),
        Some("no-device-available")
    );
}
