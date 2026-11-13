//! Tests for device enumeration, capability reporting and selection (spec
//! `opencl-backend`).

#[cfg(feature = "opencl")]
use sciencekit_common::SKGpuKernel;
use sciencekit_common::{SKBackendKind, SKError};

#[cfg(feature = "opencl")]
use super::OPENCL_KERNELS;
use super::{select_opencl_device, sk_opencl_devices, sk_register_opencl_device_inventory};

/// Enumeration is stable and every reported device is a device-backed OpenCL
/// entry. On a host with an ICD loader the list is non-empty; on a host without
/// one it is honestly empty.
#[test]
fn enumeration_is_stable_and_describes_opencl_devices() {
    let devices = sk_opencl_devices().expect("enumeration must not fail");
    let again = sk_opencl_devices().expect("enumeration must not fail twice");
    assert_eq!(devices, again, "enumeration order must be stable");

    if cfg!(feature = "opencl") {
        for device in &devices {
            assert_eq!(device.kind, SKBackendKind::OpenCl);
            assert!(device.is_device());
            assert!(device.capabilities.supports_f32 || !device.name.is_empty());
        }
        if devices.is_empty() {
            println!("[opencl] no platform reported by the ICD loader; device tests skip");
        } else {
            for device in &devices {
                println!(
                    "[opencl] device {}: `{}`, {} MiB, f64={}",
                    device.index,
                    device.name,
                    device.memory_bytes / (1024 * 1024),
                    device.capabilities.supports_f64
                );
            }
        }
    } else {
        assert!(
            devices.is_empty(),
            "the feature-off build must enumerate nothing"
        );
    }
}

/// Every OpenCL device advertises the whole kernel catalogue.
#[cfg(feature = "opencl")]
#[test]
fn every_device_advertises_the_kernel_catalogue() {
    let devices = sk_opencl_devices().expect("enumeration must not fail");
    for device in &devices {
        for kernel in OPENCL_KERNELS {
            assert!(
                device.capabilities.supports_kernel(kernel),
                "device `{}` must advertise `{}`",
                device.name,
                kernel.name()
            );
        }
    }
    assert_eq!(OPENCL_KERNELS.len(), SKGpuKernel::ALL.len());
}

/// An empty inventory produces the no-device fallback reason, never a panic and
/// never a guessed device.
#[test]
fn an_empty_inventory_has_no_device_to_select() {
    let error = select_opencl_device(&[], None, Some("f32")).expect_err("no device exists");
    match error {
        SKError::DeviceUnavailable { backend, reason } => {
            assert_eq!(backend, "opencl");
            assert!(!reason.is_empty());
        }
        other => panic!("unexpected error: {other}"),
    }
}

/// Selection is deterministic: repeated selection returns the same device, and
/// an explicit index pins exactly that device.
#[test]
fn selection_is_deterministic_and_honours_the_device_index() {
    let devices = sk_opencl_devices().expect("enumeration must not fail");
    if devices.is_empty() {
        return;
    }
    let first = select_opencl_device(&devices, None, Some("f32")).expect("a device is present");
    let second = select_opencl_device(&devices, None, Some("f32")).expect("a device is present");
    assert_eq!(first, second);

    let pinned = select_opencl_device(&devices, Some(first.index), Some("f32"))
        .expect("the selected index exists");
    assert_eq!(pinned.index, first.index);

    let missing = select_opencl_device(&devices, Some(usize::MAX), Some("f32"))
        .expect_err("an impossible index must be rejected");
    assert!(matches!(missing, SKError::DeviceUnavailable { .. }));
}

/// A scalar type the device does not execute is reported, not silently coerced.
#[test]
fn a_device_without_double_precision_reports_the_scalar_type() {
    let devices = sk_opencl_devices().expect("enumeration must not fail");
    let Some(device) = devices.first() else {
        return;
    };
    if device.capabilities.supports_f64 {
        return;
    }
    let error = select_opencl_device(&devices, Some(device.index), Some("f64"))
        .expect_err("the device lacks f64");
    assert!(matches!(error, SKError::DeviceUnavailable { .. }));
}

/// Registering the inventory makes the shared resolver pick OpenCL.
#[test]
fn registering_the_inventory_resolves_opencl_to_a_device() {
    let devices = sk_register_opencl_device_inventory().expect("registration must not fail");
    let resolved = sciencekit_common::sk_resolve_operation_backend(SKBackendKind::OpenCl);
    if devices.is_empty() {
        assert_eq!(resolved.0, sciencekit_common::SK_HOST_FALLBACK_BACKEND);
        assert_eq!(
            resolved.1,
            Some(sciencekit_common::SKFallbackReason::NoDeviceAvailable)
        );
    } else {
        assert_eq!(resolved.0, SKBackendKind::OpenCl);
        assert_eq!(resolved.1, None);
    }
    sciencekit_common::sk_clear_device_inventory();
}
