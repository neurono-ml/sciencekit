//! Deterministic device selection over an enumerated inventory (spec
//! `gpu-compute-contracts`).
//!
//! Selection is a pure function of the inventory and the caller's preference:
//! filter to the device kinds the preference accepts, sort by `(kind priority,
//! device index)` and take the first one. When nothing matches, the caller gets
//! the [`SKFallbackReason`] the plan records.

use crate::observability::SKBackendKind;

use super::fallback::SKFallbackReason;
use super::info::SKDeviceInfo;
use super::kernel::SKGpuKernel;

/// Which devices a caller accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKDevicePreference {
    /// Only a device driven by this exact backend kind.
    Backend(SKBackendKind),
    /// Any device, preferring OpenCL (vendor-transparent), then CUDA, then
    /// ROCm.
    AnyDevice,
}

/// Select the first compatible device, or report why the plan must fall back.
///
/// [`SKDevicePreference::Backend`] restricts candidates strictly to that device
/// kind; [`SKDevicePreference::AnyDevice`] accepts any device kind, ordered by
/// the vendor-transparent OpenCL kind first, then CUDA, then ROCm. Within one
/// kind, the lowest `index` wins. A host backend kind (or an empty inventory)
/// yields [`SKFallbackReason::NoDeviceAvailable`]; devices that do not support
/// `required_scalar_type` yield [`SKFallbackReason::ScalarTypeNotSupported`];
/// devices that do not implement `required_kernel` yield
/// [`SKFallbackReason::KernelNotSupported`].
pub fn sk_select_device(
    inventory: &[SKDeviceInfo],
    preference: SKDevicePreference,
    required_kernel: Option<SKGpuKernel>,
    required_scalar_type: Option<&str>,
) -> Result<SKDeviceInfo, SKFallbackReason> {
    let mut candidates: Vec<&SKDeviceInfo> = inventory
        .iter()
        .filter(|device| device.is_device())
        .filter(|device| match preference {
            SKDevicePreference::AnyDevice => true,
            SKDevicePreference::Backend(kind) => device.kind == kind,
        })
        .collect();

    if candidates.is_empty() {
        return Err(SKFallbackReason::NoDeviceAvailable);
    }

    candidates.retain(|device| match required_scalar_type {
        Some(scalar_type) => device.supports_scalar_type(scalar_type),
        None => true,
    });
    if candidates.is_empty() {
        return Err(SKFallbackReason::ScalarTypeNotSupported);
    }

    candidates.retain(|device| match required_kernel {
        Some(kernel) => device.capabilities.supports_kernel(kernel),
        None => true,
    });
    if candidates.is_empty() {
        return Err(SKFallbackReason::KernelNotSupported);
    }

    candidates.sort_by_key(|device| (kind_priority(device.kind, preference), device.index));
    candidates
        .first()
        .map(|device| (*device).clone())
        .ok_or(SKFallbackReason::NoDeviceAvailable)
}

/// The sort priority of a device kind for the given preference.
fn kind_priority(kind: SKBackendKind, preference: SKDevicePreference) -> u8 {
    if let SKDevicePreference::Backend(preferred) = preference {
        if kind == preferred {
            return 0;
        }
    }
    match kind {
        SKBackendKind::OpenCl => 1,
        SKBackendKind::Cuda => 2,
        SKBackendKind::Rocm => 3,
        // Host kinds are filtered out before sorting; the arm keeps the match
        // exhaustive and sorts them last if that invariant ever changes.
        SKBackendKind::Faer | SKBackendKind::MatrixMultiply | SKBackendKind::NdArrayLinalg => 4,
    }
}
