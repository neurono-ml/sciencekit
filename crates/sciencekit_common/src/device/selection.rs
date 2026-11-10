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
/// The ordering is deterministic: devices of the preferred kind come first,
/// then devices of the vendor-transparent OpenCL kind, then CUDA, then ROCm;
/// within one kind the lowest device index wins. A host backend kind (or an
/// empty inventory) yields [`SKFallbackReason::NoDeviceAvailable`].
pub fn sk_select_device(
    inventory: &[SKDeviceInfo],
    preference: SKDevicePreference,
) -> Result<SKDeviceInfo, SKFallbackReason> {
    let mut candidates: Vec<&SKDeviceInfo> = inventory
        .iter()
        .filter(|device| device.is_device())
        .filter(|device| match preference {
            SKDevicePreference::AnyDevice => true,
            SKDevicePreference::Backend(kind) => device.kind == kind,
        })
        .collect();

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
