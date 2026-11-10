//! The host↔device transfer policy (spec `gpu-compute-contracts`).

/// How a backend moves data between host and device.
///
/// The safe default is an explicit copy: uploads and downloads are visible
/// costs the caller can reason about. A mapped (zero-copy) view is opt-in and
/// only available when [`SKDeviceCapabilities::supports_mapped_memory`](sciencekit_common::SKDeviceCapabilities::supports_mapped_memory)
/// is set for the selected device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SKDeviceTransferPolicy {
    /// Copy host data into device storage and back (the default).
    #[default]
    Explicit,
    /// Map device storage into the host address space when the device supports
    /// it, falling back to an explicit copy otherwise.
    MappedWhenSupported,
}
