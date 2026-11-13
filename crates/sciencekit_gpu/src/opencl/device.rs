//! OpenCL device enumeration, capability reporting and deterministic selection
//! (spec `opencl-backend`, requirement *OpenCL device discovery*).
//!
//! Enumeration is explicit and never performed at crate load. With the
//! `opencl` feature off the enumeration is empty without calling any OpenCL
//! symbol; with the feature on it walks the platforms the ICD loader exposes and
//! reports what each device advertises. Selection reuses the shared, pure
//! [`sk_select_device`] so OpenCL obeys exactly the same ordering rules as every
//! other backend.

#[cfg(feature = "opencl")]
use sciencekit_common::SKDeviceCapabilities;
#[cfg(feature = "opencl")]
use sciencekit_common::SKGpuKernel;
use sciencekit_common::{
    SKBackendKind, SKDeviceInfo, SKDevicePreference, SKError, sk_register_device_inventory,
    sk_select_device,
};

use super::availability::{feature_disabled_reason, unavailable_error};

/// The kernels the OpenCL implementation provides on every compatible device.
#[cfg(feature = "opencl")]
pub(crate) const OPENCL_KERNELS: [SKGpuKernel; 4] = SKGpuKernel::ALL;

/// Enumerate the OpenCL devices the ICD loader reports.
///
/// The order is the loader's platform order, then each platform's device order,
/// so the returned index is stable across calls on the same host. With the
/// feature off this returns an empty list and never touches a runtime symbol.
pub fn sk_opencl_devices() -> Result<Vec<SKDeviceInfo>, SKError> {
    #[cfg(feature = "opencl")]
    {
        enumerate_devices()
    }
    #[cfg(not(feature = "opencl"))]
    {
        Ok(Vec::new())
    }
}

/// Enumerate the OpenCL devices and register them as the process-wide inventory.
///
/// After this call [`sk_resolve_operation_backend`](sciencekit_common::sk_resolve_operation_backend)
/// resolves [`SKBackendKind::OpenCl`] to a device instead of the CPU fallback.
/// With the feature off the inventory stays empty, which is the honest answer.
pub fn sk_register_opencl_device_inventory() -> Result<Vec<SKDeviceInfo>, SKError> {
    let devices = sk_opencl_devices()?;
    sk_register_device_inventory(devices.clone());
    Ok(devices)
}

/// Select the device a backend should drive.
///
/// `required_scalar_type` lets the shared selector reject a device that lacks
/// the caller's scalar precision, and an explicit `device_index` pins one
/// enumerated device. Both failures are reported, never guessed around.
pub(crate) fn select_opencl_device(
    devices: &[SKDeviceInfo],
    device_index: Option<usize>,
    required_scalar_type: Option<&str>,
) -> Result<SKDeviceInfo, SKError> {
    if devices.is_empty() {
        return Err(unavailable_error(if cfg!(feature = "opencl") {
            "no OpenCL platform or device is installed".to_string()
        } else {
            feature_disabled_reason()
        }));
    }
    let selection = sk_select_device(
        devices,
        SKDevicePreference::Backend(SKBackendKind::OpenCl),
        None,
        required_scalar_type,
    )
    .map_err(|reason| unavailable_error(format!("no OpenCL device matches: {reason}")))?;
    match device_index {
        Some(index) => devices
            .iter()
            .find(|device| device.index == index)
            .cloned()
            .ok_or_else(|| unavailable_error(format!("no OpenCL device with index {index}"))),
        None => Ok(selection),
    }
}

/// Build one device description from the properties opencl3 reports.
#[cfg(feature = "opencl")]
pub(crate) fn describe_device(index: usize, device: &opencl3::device::Device) -> SKDeviceInfo {
    let name = device
        .name()
        .unwrap_or_else(|_| format!("OpenCL device {index}"));
    let memory_bytes = device.global_mem_size().unwrap_or(0);
    let supports_f64 = device
        .double_fp_config()
        .map(|config| config != 0)
        .unwrap_or(false);
    let supports_mapped_memory = device.host_unified_memory().unwrap_or(false);
    SKDeviceInfo::new(
        index,
        SKBackendKind::OpenCl,
        name,
        memory_bytes,
        SKDeviceCapabilities::new(true, supports_f64, supports_mapped_memory)
            .with_kernels(&OPENCL_KERNELS),
    )
}

/// Enumerate the device identifiers in the same order as the descriptions.
#[cfg(feature = "opencl")]
pub(crate) fn device_identifiers() -> Result<Vec<(usize, opencl3::types::cl_device_id)>, SKError> {
    use opencl3::device::{CL_DEVICE_TYPE_ALL, get_all_devices};

    let identifiers = get_all_devices(CL_DEVICE_TYPE_ALL).map_err(|code| {
        unavailable_error(format!("OpenCL device enumeration failed with code {code}"))
    })?;
    Ok(identifiers.into_iter().enumerate().collect())
}

#[cfg(feature = "opencl")]
fn enumerate_devices() -> Result<Vec<SKDeviceInfo>, SKError> {
    use opencl3::device::Device;

    let mut devices = Vec::new();
    for (index, identifier) in device_identifiers()? {
        devices.push(describe_device(index, &Device::new(identifier)));
    }
    Ok(devices)
}

#[cfg(test)]
mod device_tests;
