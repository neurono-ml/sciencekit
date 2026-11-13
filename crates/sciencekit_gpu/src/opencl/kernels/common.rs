//! Shared kernel-launch plumbing (spec `opencl-backend`, requirement *OpenCL
//! kernels*).
//!
//! Every kernel follows the same shape: compile (or reuse) the program for the
//! caller's scalar type, create the kernel object, set its arguments, enqueue a
//! work range and let the in-order command queue guarantee that the following
//! blocking read sees the result. This module owns the compile step so each
//! kernel file stays about its mathematics.

use opencl3::kernel::Kernel;
use sciencekit_common::SKError;

use super::source::scalar_source;
use crate::opencl::session::SKOpenClSession;

/// Compile (or reuse) the program for `template` and create `kernel_name`.
pub(crate) fn create_kernel(
    session: &SKOpenClSession,
    program_key: &'static str,
    template: &str,
    kernel_name: &'static str,
    scalar_type: &str,
) -> Result<Kernel, SKError> {
    let source = scalar_source(template, scalar_type)?;
    let program = session.program(program_key, &source)?;
    Kernel::create(&program, kernel_name).map_err(|code| {
        SKError::Conversion(format!(
            "OpenCL kernel `{kernel_name}` creation failed with code {code}"
        ))
    })
}
