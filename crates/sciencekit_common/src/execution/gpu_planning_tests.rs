//! Tests for GPU-aware execution planning (spec `gpu-compute-contracts`).

use crate::{
    SKBackendKind, SKDeviceCapabilities, SKDeviceContext, SKDeviceInfo, SKFallbackReason,
    SKGpuKernel,
};

use super::{SKAccessPattern, SKExecutionContext, SKExecutionMode, sk_resolve_execution_plan};

fn device(kind: SKBackendKind, kernels: &[SKGpuKernel]) -> SKDeviceInfo {
    SKDeviceInfo::new(
        0,
        kind,
        "Mock Device 0",
        1 << 30,
        SKDeviceCapabilities::new(true, true, false).with_kernels(kernels),
    )
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

fn device_context(
    configured_backend: SKBackendKind,
    required_kernel: Option<SKGpuKernel>,
    inventory: Vec<SKDeviceInfo>,
) -> SKDeviceContext {
    SKDeviceContext {
        configured_backend,
        required_kernel,
        device_inventory: inventory,
    }
}

/// An automatic plan selects the configured device when it is available.
#[test]
fn automatic_plan_selects_the_configured_device_when_available() {
    let context = context(Some(device_context(
        SKBackendKind::OpenCl,
        Some(SKGpuKernel::Gemm),
        vec![device(SKBackendKind::OpenCl, &[SKGpuKernel::Gemm])],
    )));

    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap();
    assert_eq!(plan.backend, SKBackendKind::OpenCl);
    assert_eq!(plan.fallback_reason, None);
}

/// A requested device that is not enumerated falls back to the host backend.
#[test]
fn automatic_plan_falls_back_to_host_without_a_device() {
    let context = context(Some(device_context(SKBackendKind::Cuda, None, Vec::new())));

    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap();
    assert_eq!(plan.backend, SKBackendKind::Faer);
    assert_eq!(
        plan.fallback_reason,
        Some(SKFallbackReason::NoDeviceAvailable)
    );
}

/// A device that lacks the required kernel is not selected.
#[test]
fn automatic_plan_falls_back_when_the_kernel_is_missing() {
    let context = context(Some(device_context(
        SKBackendKind::OpenCl,
        Some(SKGpuKernel::TreePredict),
        vec![device(SKBackendKind::OpenCl, &[SKGpuKernel::Elementwise])],
    )));

    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap();
    assert_eq!(plan.backend, SKBackendKind::Faer);
    assert_eq!(
        plan.fallback_reason,
        Some(SKFallbackReason::KernelNotSupported)
    );
}

/// A configured host backend stays on the host with no fallback reason.
#[test]
fn automatic_plan_keeps_the_configured_host_backend() {
    let context = context(Some(device_context(
        SKBackendKind::MatrixMultiply,
        None,
        vec![device(SKBackendKind::OpenCl, &[SKGpuKernel::Gemm])],
    )));

    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap();
    assert_eq!(plan.backend, SKBackendKind::MatrixMultiply);
    assert_eq!(plan.fallback_reason, None);
}

/// A context without a device inventory plans exactly as before the change.
#[test]
fn a_context_without_devices_plans_on_the_default_host_backend() {
    let plan = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context(None)).unwrap();
    assert_eq!(plan.backend, SKBackendKind::Faer);
    assert_eq!(plan.fallback_reason, None);
}

/// An explicit execution mode never selects a device.
#[test]
fn an_explicit_mode_never_selects_a_device() {
    let context = context(Some(device_context(
        SKBackendKind::OpenCl,
        None,
        vec![device(SKBackendKind::OpenCl, &[SKGpuKernel::Gemm])],
    )));

    let plan = sk_resolve_execution_plan(SKExecutionMode::InProcessSynchronous, &context).unwrap();
    assert_eq!(plan.backend, SKBackendKind::Faer);
    assert_eq!(
        plan.fallback_reason,
        Some(SKFallbackReason::ExplicitModeDoesNotSelectDevice)
    );
}

/// The same context always resolves to the same backend decision.
#[test]
fn gpu_planning_is_deterministic() {
    let context = context(Some(device_context(
        SKBackendKind::OpenCl,
        Some(SKGpuKernel::PairwiseDistance),
        vec![
            device(SKBackendKind::OpenCl, &[SKGpuKernel::PairwiseDistance]),
            device(SKBackendKind::Cuda, &[SKGpuKernel::PairwiseDistance]),
        ],
    )));

    let first = sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap();
    for _ in 0..8 {
        assert_eq!(
            sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap(),
            first
        );
    }
}
