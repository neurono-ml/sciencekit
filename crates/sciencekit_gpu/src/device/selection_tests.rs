//! Tests for deterministic device selection (spec `gpu-compute-contracts`).
//!
//! The tests exercise the public contract re-exported by this crate, which is
//! implemented by `sciencekit_common` so execution planning can select a device
//! without depending on this crate.

use sciencekit_common::{
    SK_HOST_FALLBACK_BACKEND, SKBackendKind, SKDeviceCapabilities, SKDeviceInfo,
    SKDevicePreference, SKFallbackReason, sk_select_device,
};

fn device(index: usize, kind: SKBackendKind, name: &'static str) -> SKDeviceInfo {
    SKDeviceInfo::new(
        index,
        kind,
        name,
        1024,
        SKDeviceCapabilities::new(true, true, false),
    )
}

/// Several devices are ordered by kind priority, then by device index.
#[test]
fn multiple_devices_are_ordered_by_kind_priority_then_index() {
    // Inventory deliberately out of priority order: CUDA before OpenCL, and
    // OpenCL device 1 before device 0.
    let inventory = vec![
        device(0, SKBackendKind::Cuda, "CUDA 0"),
        device(1, SKBackendKind::OpenCl, "OpenCL 1"),
        device(0, SKBackendKind::OpenCl, "OpenCL 0"),
        device(0, SKBackendKind::Rocm, "ROCm 0"),
    ];

    let selected = sk_select_device(&inventory, SKDevicePreference::AnyDevice, None)
        .expect("a device is present");
    assert_eq!(selected.name, "OpenCL 0");
}

/// A configured backend selects its own first device, ahead of a
/// higher-priority kind.
#[test]
fn a_configured_backend_selects_its_first_device() {
    let inventory = vec![
        device(0, SKBackendKind::OpenCl, "OpenCL 0"),
        device(1, SKBackendKind::Cuda, "CUDA 1"),
        device(0, SKBackendKind::Cuda, "CUDA 0"),
    ];

    let selected = sk_select_device(
        &inventory,
        SKDevicePreference::Backend(SKBackendKind::Cuda),
        None,
    )
    .expect("a CUDA device is present");
    assert_eq!(selected.name, "CUDA 0");
}

/// With no matching device, selection reports the CPU fallback reason.
#[test]
fn no_matching_device_reports_the_host_fallback() {
    let inventory = vec![device(0, SKBackendKind::Cuda, "CUDA 0")];

    let reason = sk_select_device(
        &inventory,
        SKDevicePreference::Backend(SKBackendKind::OpenCl),
        None,
    )
    .expect_err("no OpenCL device is present");
    assert_eq!(reason, SKFallbackReason::NoDeviceAvailable);
    assert_eq!(SK_HOST_FALLBACK_BACKEND, SKBackendKind::Faer);
}

/// An empty inventory falls back without an error.
#[test]
fn an_empty_inventory_falls_back_to_the_host_backend() {
    let reason = sk_select_device(&[], SKDevicePreference::AnyDevice, None)
        .expect_err("no device is present");
    assert_eq!(reason, SKFallbackReason::NoDeviceAvailable);
}

/// A host backend preference never selects a device.
#[test]
fn a_host_backend_preference_never_selects_a_device() {
    let inventory = vec![device(0, SKBackendKind::OpenCl, "OpenCL 0")];
    let reason = sk_select_device(
        &inventory,
        SKDevicePreference::Backend(SKBackendKind::Faer),
        None,
    )
    .expect_err("a host backend is not a device");
    assert_eq!(reason, SKFallbackReason::NoDeviceAvailable);
}

/// Repeated selection over the same inventory yields the same device.
#[test]
fn selection_is_deterministic_across_repeated_calls() {
    let inventory = vec![
        device(1, SKBackendKind::Rocm, "ROCm 1"),
        device(1, SKBackendKind::OpenCl, "OpenCL 1"),
        device(0, SKBackendKind::OpenCl, "OpenCL 0"),
    ];
    let first = sk_select_device(&inventory, SKDevicePreference::AnyDevice, None).unwrap();
    for _ in 0..16 {
        assert_eq!(
            sk_select_device(&inventory, SKDevicePreference::AnyDevice, None).unwrap(),
            first
        );
    }
}

/// Every fallback reason renders as a stable kebab-case token for spans.
#[test]
fn fallback_reasons_render_as_stable_tokens() {
    assert_eq!(
        SKFallbackReason::NoDeviceAvailable.to_string(),
        "no-device-available"
    );
    assert_eq!(
        SKFallbackReason::KernelNotSupported.to_string(),
        "kernel-not-supported"
    );
    assert_eq!(
        SKFallbackReason::ExplicitModeDoesNotSelectDevice.to_string(),
        "explicit-mode-does-not-select-device"
    );
}
