//! The availability-detection abstraction every device backend implements
//! (spec `gpu-compute-contracts`).

use sciencekit_common::{SKDeviceInfo, SKError};

/// Enumerate the devices a backend can drive.
///
/// Discovery is explicit — never performed at crate load — and returns an empty
/// list when no device is present. The order is the backend's stable
/// enumeration order, and each device carries the [`SKDeviceInfo::index`] the
/// planner sorts by. Concrete backends may cache the result; planning itself
/// stays pure because it receives the inventory as input.
///
/// The error type converts from [`SKError`], so a backend reports a structured
/// failure (an unavailable driver, a denied device node) without leaking a
/// runtime-specific type into planning.
pub trait SKDeviceDiscovery: Send + Sync {
    /// The backend's structured error type.
    type Error: std::error::Error + From<SKError> + Send + Sync + 'static;

    /// Enumerate the devices the backend can drive, in its stable order.
    fn discover_devices(&self) -> Result<Vec<SKDeviceInfo>, Self::Error>;
}
