//! The kernel-dispatch surface (spec `gpu-compute-contracts`).
//!
//! A backend receives the kernel and the argument buffers it operates on.
//! Dispatch is explicit — the caller uploaded the inputs and downloads the
//! result — so transfer costs stay visible at the call site.

use crate::buffer::SKDeviceBuffer;

/// A borrowed argument buffer passed to
/// [`SKComputeBackend::dispatch`](crate::backend::SKComputeBackend::dispatch).
pub type SKDeviceBufferRef<'a, F> = &'a SKDeviceBuffer<F>;
