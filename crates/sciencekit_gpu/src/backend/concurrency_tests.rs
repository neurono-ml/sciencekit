//! Concurrency tests for device discovery and planning (spec
//! `gpu-compute-contracts`).
//!
//! Discovery and planning are shared across threads by construction: both are
//! `Send + Sync` and neither mutates shared state, so concurrent calls must
//! agree exactly.

use std::sync::Arc;
use std::thread;

use ndarray::array;
use sciencekit_common::execution::{SKAccessPattern, SKExecutionMode, sk_resolve_execution_plan};
use sciencekit_common::{
    SKBackendKind, SKDeviceCapabilities, SKDeviceContext, SKDeviceInfo, SKError,
    SKExecutionContext, SKExecutionPlan, SKFallbackReason, SKGpuKernel,
};

use crate::backend::mock_device::SKMockComputeBackend;
use crate::backend::{SKComputeBackend, SKDeviceDiscovery};
use crate::buffer::SKDeviceBuffer;

const THREADS: usize = 8;

fn assert_send_and_sync<T: Send + Sync + ?Sized>() {}

/// The contract types are safe to share across threads.
#[test]
fn contract_types_are_send_and_sync() {
    assert_send_and_sync::<SKDeviceInfo>();
    assert_send_and_sync::<SKDeviceCapabilities>();
    assert_send_and_sync::<SKDeviceContext>();
    assert_send_and_sync::<SKExecutionContext>();
    assert_send_and_sync::<SKExecutionPlan>();
    assert_send_and_sync::<SKGpuKernel>();
    assert_send_and_sync::<SKFallbackReason>();
    assert_send_and_sync::<SKDeviceBuffer<f64>>();
    assert_send_and_sync::<SKMockComputeBackend>();
    assert_send_and_sync::<dyn SKComputeBackend<f64, Error = SKError>>();
}

fn backend() -> SKMockComputeBackend {
    SKMockComputeBackend::new(vec![
        SKMockComputeBackend::device(
            0,
            SKBackendKind::OpenCl,
            "Mock OpenCL 0",
            1 << 20,
            SKDeviceCapabilities::new(true, true, false).with_kernels(&[SKGpuKernel::Gemm]),
        ),
        SKMockComputeBackend::device(
            0,
            SKBackendKind::Cuda,
            "Mock CUDA 0",
            1 << 20,
            SKDeviceCapabilities::new(true, true, false).with_kernels(&[SKGpuKernel::Gemm]),
        ),
    ])
}

/// Concurrent discovery returns the same inventory on every thread.
#[test]
fn concurrent_discovery_agrees() {
    let backend = Arc::new(backend());
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let backend = Arc::clone(&backend);
            thread::spawn(move || backend.discover_devices().unwrap())
        })
        .collect();

    let inventories: Vec<Vec<SKDeviceInfo>> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    let first = &inventories[0];
    assert_eq!(first.len(), 2);
    assert!(inventories.iter().all(|inventory| inventory == first));
}

/// Concurrent planning over one shared context yields identical plans.
#[test]
fn concurrent_planning_agrees() {
    let inventory = backend().discover_devices().unwrap();
    assert_eq!(inventory.len(), 2);
    let context = Arc::new(SKExecutionContext {
        available_memory_bytes: 1 << 30,
        cpu_cores: 4,
        dataset_size_bytes: 1024,
        dataset_elements: Some(1024),
        scalar_size_bytes: 8,
        grain: 256,
        access_pattern: SKAccessPattern::Sequential,
        batch_size_hint: None,
        batch_byte_size_hint: None,
        device: Some(
            SKDeviceContext::new(SKBackendKind::OpenCl)
                .with_required_kernel(SKGpuKernel::Gemm)
                .with_inventory(inventory),
        ),
    });

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let context = Arc::clone(&context);
            thread::spawn(move || {
                sk_resolve_execution_plan(SKExecutionMode::Automatic, &context).unwrap()
            })
        })
        .collect();

    let plans: Vec<SKExecutionPlan> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    let first = &plans[0];
    assert_eq!(first.backend, SKBackendKind::OpenCl);
    assert_eq!(first.fallback_reason, None);
    assert!(plans.iter().all(|plan| plan == first));
}

/// Concurrent uploads and downloads over one shared backend agree.
#[test]
fn concurrent_transfers_agree() {
    let backend = Arc::new(backend());
    let host = array![[1.0_f64, 2.0], [3.0, 4.0]];

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let backend = Arc::clone(&backend);
            let host = host.clone();
            thread::spawn(move || {
                let buffer = backend.upload(host.view()).unwrap();
                backend.download(&buffer).unwrap()
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().unwrap(), host);
    }
}
