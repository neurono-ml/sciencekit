//! Tests for the kernel catalogue and dispatch contract (spec
//! `gpu-compute-contracts`).

use std::str::FromStr;

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKDeviceCapabilities, SKDeviceInfo, SKDevicePreference, SKError,
    SKFallbackReason, SKGpuKernel, sk_select_device,
};

use crate::backend::SKComputeBackend;
use crate::backend::mock_device::SKMockComputeBackend;
use crate::kernel::SKDeviceBufferRef;

fn capabilities(kernels: &[SKGpuKernel]) -> SKDeviceCapabilities {
    SKDeviceCapabilities::new(true, true, false).with_kernels(kernels)
}

fn backend_with(kernels: &[SKGpuKernel]) -> SKMockComputeBackend {
    SKMockComputeBackend::new(vec![SKMockComputeBackend::device(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1 << 20,
        capabilities(kernels),
    )])
}

/// The catalogue is closed and names every kernel exactly once.
#[test]
fn the_kernel_catalogue_lists_every_kernel() {
    let names: Vec<&str> = SKGpuKernel::ALL
        .iter()
        .map(|kernel| kernel.name())
        .collect();
    assert_eq!(
        names,
        vec!["pairwise-distance", "gemm", "elementwise", "tree-predict"]
    );
    for kernel in SKGpuKernel::ALL {
        assert_eq!(SKGpuKernel::from_str(kernel.name()).unwrap(), kernel);
    }
}

/// A supported kernel dispatches and returns a device buffer.
#[test]
fn a_supported_kernel_dispatches_and_returns_a_buffer() {
    let backend = backend_with(&[SKGpuKernel::Elementwise]);
    let input = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let buffer = backend.upload(input.view()).expect("upload must succeed");

    let output = backend
        .dispatch(SKGpuKernel::Elementwise, &[&buffer])
        .expect("the kernel is supported");
    assert_eq!(output.shape(), (2, 2));
    let host = backend.download(&output).expect("download must succeed");
    assert_eq!(host, input);
}

/// An unsupported kernel is reported by name instead of running silently.
#[test]
fn an_unsupported_kernel_names_itself_in_the_error() {
    let backend = backend_with(&[SKGpuKernel::Elementwise]);
    let input = array![[1.0_f64]];
    let buffer = backend.upload(input.view()).expect("upload must succeed");

    let error = backend
        .dispatch(SKGpuKernel::Gemm, &[&buffer])
        .expect_err("GEMM is not supported by the mock");
    match error {
        SKError::UnsupportedOperation { operation, reason } => {
            assert_eq!(operation, "gemm");
            assert!(reason.contains("Mock OpenCL 0"));
        }
        other => panic!("expected `UnsupportedOperation`, got {other:?}"),
    }
}

/// Dispatching with no argument buffer is a structured failure.
#[test]
fn dispatching_without_arguments_is_rejected() {
    let backend = backend_with(&[SKGpuKernel::Elementwise]);
    let error = backend
        .dispatch(
            SKGpuKernel::Elementwise,
            &[] as &[SKDeviceBufferRef<'_, f64>],
        )
        .expect_err("dispatch needs an input buffer");
    assert!(matches!(error, SKError::Conversion(_)));
}

/// A device without the required kernel is filtered out by selection.
#[test]
fn a_device_without_the_required_kernel_is_not_selected() {
    let inventory = vec![SKDeviceInfo::new(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1024,
        capabilities(&[SKGpuKernel::Elementwise]),
    )];
    let reason = sk_select_device(
        &inventory,
        SKDevicePreference::AnyDevice,
        Some(SKGpuKernel::Gemm),
        None,
    )
    .expect_err("GEMM is not supported");
    assert_eq!(reason, SKFallbackReason::KernelNotSupported);
}

/// A device carrying the required kernel is selected.
#[test]
fn a_device_with_the_required_kernel_is_selected() {
    let inventory = vec![SKDeviceInfo::new(
        0,
        SKBackendKind::OpenCl,
        "Mock OpenCL 0",
        1024,
        capabilities(&[SKGpuKernel::Gemm]),
    )];
    let device = sk_select_device(
        &inventory,
        SKDevicePreference::AnyDevice,
        Some(SKGpuKernel::Gemm),
        None,
    )
    .expect("GEMM is supported");
    assert_eq!(device.name, "Mock OpenCL 0");
}

/// When device 0 lacks the required kernel and device 1 has it, selection picks device 1.
#[test]
fn selection_skips_device_missing_kernel_and_picks_second_device() {
    let inventory = vec![
        SKDeviceInfo::new(
            0,
            SKBackendKind::OpenCl,
            "Mock OpenCL 0",
            1024,
            capabilities(&[SKGpuKernel::Elementwise]),
        ),
        SKDeviceInfo::new(
            1,
            SKBackendKind::OpenCl,
            "Mock OpenCL 1",
            1024,
            capabilities(&[SKGpuKernel::Gemm]),
        ),
    ];
    let device = sk_select_device(
        &inventory,
        SKDevicePreference::AnyDevice,
        Some(SKGpuKernel::Gemm),
        Some("f64"),
    )
    .expect("device 1 implements GEMM");
    assert_eq!(device.index, 1);
    assert_eq!(device.name, "Mock OpenCL 1");
}
