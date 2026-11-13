//! The builder for [`SKOpenClBackend`](super::SKOpenClBackend) (spec
//! `opencl-backend`, requirement *OpenCL device discovery*).
//!
//! The builder is the only public constructor. It selects the device
//! deterministically through the shared selector, opens the session lazily, and
//! records why the backend is unavailable instead of panicking. The `backend`
//! selection itself is not a builder knob here: an OpenCL backend is always
//! [`SKBackendKind::OpenCl`](sciencekit_common::SKBackendKind::OpenCl) by
//! construction and every operation records that kind.

use sciencekit_common::SKError;

use super::implementation::SKOpenClBackend;
use crate::buffer::SKDeviceTransferPolicy;
use crate::opencl::availability::{feature_disabled_reason, is_feature_enabled};
use crate::opencl::device::{select_opencl_device, sk_opencl_devices};

/// Builds an [`SKOpenClBackend`].
#[derive(Debug, Clone)]
pub struct SKOpenClBackendBuilder {
    device_index: Option<usize>,
    transfer_policy: SKDeviceTransferPolicy,
}

impl SKOpenClBackendBuilder {
    /// Start a builder with the default device selection.
    pub fn new() -> Self {
        SKOpenClBackendBuilder {
            device_index: None,
            transfer_policy: SKDeviceTransferPolicy::Explicit,
        }
    }

    /// Pin the enumerated device index the backend must drive.
    pub fn device_index(mut self, index: usize) -> Self {
        self.device_index = Some(index);
        self
    }

    /// Set the host↔device transfer policy.
    pub fn transfer_policy(mut self, policy: SKDeviceTransferPolicy) -> Self {
        self.transfer_policy = policy;
        self
    }

    /// Build the backend, failing when no device can serve the requested index.
    pub fn build(self) -> Result<SKOpenClBackend, SKError> {
        let devices = sk_opencl_devices()?;
        let device = select_opencl_device(&devices, self.device_index, None)?;
        SKOpenClBackend::open(device, self.transfer_policy)
    }

    /// Build the backend, recording the reason instead of failing.
    ///
    /// This is the path planning takes: the returned backend reports
    /// [`SKFallbackReason::NoDeviceAvailable`](sciencekit_common::SKFallbackReason::NoDeviceAvailable)
    /// and every operation returns a structured error, so the caller can fall
    /// back to the CPU with an honest reason.
    pub fn build_or_fallback(self) -> SKOpenClBackend {
        let transfer_policy = self.transfer_policy;
        match self.build() {
            Ok(backend) => backend,
            Err(error) => {
                let reason = if is_feature_enabled() {
                    error.to_string()
                } else {
                    feature_disabled_reason()
                };
                SKOpenClBackend::unavailable(reason, transfer_policy)
            }
        }
    }
}

impl Default for SKOpenClBackendBuilder {
    fn default() -> Self {
        SKOpenClBackendBuilder::new()
    }
}
