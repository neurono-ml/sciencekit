//! The host↔device transfer policy (spec `gpu-compute-contracts`).

/// How a backend moves data between host and device.
///
/// Every [`SKComputeBackend`](crate::SKComputeBackend) declares its policy via
/// [`SKComputeBackend::transfer_policy`](crate::SKComputeBackend::transfer_policy),
/// defaulting to [`SKDeviceTransferPolicy::Explicit`]: uploads and downloads are
/// visible copies the caller can reason about. A backend that returns
/// [`SKDeviceTransferPolicy::MappedWhenSupported`] maps device storage into the
/// host address space when the selected device advertises
/// [`SKDeviceCapabilities::supports_mapped_memory`](sciencekit_common::SKDeviceCapabilities::supports_mapped_memory).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SKDeviceTransferPolicy {
    /// Copy host data into device storage and back (the default).
    #[default]
    Explicit,
    /// Use mapped (zero-copy) host↔device storage when the selected device
    /// advertises mapped-memory support.
    MappedWhenSupported,
}
