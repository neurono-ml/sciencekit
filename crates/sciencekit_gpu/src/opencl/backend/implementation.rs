//! The concrete OpenCL backend (spec `opencl-backend`).
//!
//! The backend owns the selected device session, implements the shared
//! [`SKComputeBackend`] and [`SKDeviceDiscovery`] contracts, and reports an
//! honest fallback when the feature is off or no device is present. Every
//! public operation opens an observability span that records
//! [`SKBackendKind::OpenCl`] — never a hardcoded kind string at the call site.

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "opencl")]
use std::sync::Arc;

use sciencekit_common::{
    SK_HOST_FALLBACK_BACKEND, SKBackendKind, SKDeviceInfo, SKError, SKFallbackReason,
};

use super::builder::SKOpenClBackendBuilder;
use crate::buffer::SKDeviceTransferPolicy;
use crate::opencl::availability::{fallback_reason as recorded_fallback_reason, unavailable_error};

#[cfg(feature = "opencl")]
use crate::opencl::session::SKOpenClSession;

/// An accounting report of the bytes this backend moved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SKOpenClTransferReport {
    /// Bytes copied host → device by uploads.
    pub upload_bytes: u64,
    /// Bytes copied device → host by downloads.
    pub download_bytes: u64,
    /// Bytes read back from kernel outputs.
    pub kernel_bytes: u64,
}

impl SKOpenClTransferReport {
    /// The total bytes the backend moved.
    pub fn total_bytes(&self) -> u64 {
        self.upload_bytes + self.download_bytes + self.kernel_bytes
    }
}

/// The OpenCL compute backend.
///
/// Construct it with [`SKOpenClBackend::builder`]; the constructor is private.
/// A backend whose device could not be opened still exists — it reports
/// [`SKOpenClBackend::fallback_reason`] and returns structured errors — so
/// planning can fall back to the CPU with an honest reason.
pub struct SKOpenClBackend {
    #[cfg(feature = "opencl")]
    session: Option<Arc<SKOpenClSession>>,
    unavailable_reason: String,
    pub(crate) transfer_policy: SKDeviceTransferPolicy,
    pub(crate) upload_bytes: AtomicU64,
    pub(crate) download_bytes: AtomicU64,
    pub(crate) kernel_bytes: AtomicU64,
}

impl SKOpenClBackend {
    /// Start building an OpenCL backend.
    pub fn builder() -> SKOpenClBackendBuilder {
        SKOpenClBackendBuilder::new()
    }

    /// Open the backend on a selected device (the only success constructor).
    #[cfg(feature = "opencl")]
    pub(crate) fn open(
        device: SKDeviceInfo,
        transfer_policy: SKDeviceTransferPolicy,
    ) -> Result<Self, SKError> {
        let session = SKOpenClSession::open(device.index)?;
        Ok(SKOpenClBackend {
            session: Some(Arc::new(session)),
            unavailable_reason: String::new(),
            transfer_policy,
            upload_bytes: AtomicU64::new(0),
            download_bytes: AtomicU64::new(0),
            kernel_bytes: AtomicU64::new(0),
        })
    }

    /// Build the fallback backend when the feature is off.
    #[cfg(not(feature = "opencl"))]
    pub(crate) fn open(
        _device: SKDeviceInfo,
        _transfer_policy: SKDeviceTransferPolicy,
    ) -> Result<Self, SKError> {
        Err(unavailable_error(
            "the `opencl` feature of `sciencekit_gpu` is disabled; CPU execution runs instead",
        ))
    }

    /// Build a backend that reports `reason` for every operation.
    pub(crate) fn unavailable(reason: String, transfer_policy: SKDeviceTransferPolicy) -> Self {
        SKOpenClBackend {
            #[cfg(feature = "opencl")]
            session: None,
            unavailable_reason: reason,
            transfer_policy,
            upload_bytes: AtomicU64::new(0),
            download_bytes: AtomicU64::new(0),
            kernel_bytes: AtomicU64::new(0),
        }
    }

    /// The selected device, when a device was opened.
    pub fn device(&self) -> Option<&SKDeviceInfo> {
        #[cfg(feature = "opencl")]
        {
            self.session.as_ref().map(|session| session.device())
        }
        #[cfg(not(feature = "opencl"))]
        {
            None
        }
    }

    /// Whether the backend can serve operations.
    pub fn is_available(&self) -> bool {
        self.device().is_some()
    }

    /// Ok when a device was opened, the structured reason otherwise.
    pub fn availability(&self) -> Result<(), SKError> {
        match self.is_available() {
            true => Ok(()),
            false => Err(self.unavailable_error()),
        }
    }

    /// Why planning must fall back to the CPU, or `None` when a device is ready.
    pub fn fallback_reason(&self) -> Option<SKFallbackReason> {
        if self.is_available() {
            None
        } else {
            Some(recorded_fallback_reason())
        }
    }

    /// Open the span for one public operation.
    ///
    /// An available backend records [`SKBackendKind::OpenCl`]; an unavailable
    /// one records the host fallback kind and its reason, so the span never
    /// claims device work that did not happen.
    pub(crate) fn operation_span(&self, operation: &'static str) -> tracing::Span {
        let (backend, reason) = if self.is_available() {
            (SKBackendKind::OpenCl, None)
        } else {
            (SK_HOST_FALLBACK_BACKEND, self.fallback_reason())
        };
        crate::opencl::observability::operation_span(operation, backend, reason)
    }

    /// The bytes this backend has moved so far.
    pub fn transfer_report(&self) -> SKOpenClTransferReport {
        SKOpenClTransferReport {
            upload_bytes: self.upload_bytes.load(Ordering::Relaxed),
            download_bytes: self.download_bytes.load(Ordering::Relaxed),
            kernel_bytes: self.kernel_bytes.load(Ordering::Relaxed),
        }
    }

    /// Enumerate the devices this host exposes and register them process-wide.
    pub fn register_inventory(&self) -> Result<Vec<SKDeviceInfo>, SKError> {
        crate::opencl::device::sk_register_opencl_device_inventory()
    }

    /// The session, or the structured unavailability error.
    #[cfg(feature = "opencl")]
    pub(crate) fn session(&self) -> Result<&SKOpenClSession, SKError> {
        self.session
            .as_ref()
            .map(|session| session.as_ref())
            .ok_or_else(|| unavailable_error(self.unavailable_reason.clone()))
    }

    /// The error every operation reports when no device was opened.
    pub(crate) fn unavailable_error(&self) -> SKError {
        unavailable_error(self.unavailable_reason.clone())
    }
}

impl std::fmt::Debug for SKOpenClBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SKOpenClBackend")
            .field("device", &self.device().map(|device| device.name.clone()))
            .field("transfer_policy", &self.transfer_policy)
            .finish()
    }
}
