//! The live OpenCL session: one context, one command queue and the cached
//! programs of the selected device.
//!
//! A session is created only when the `opencl` feature is on and a device was
//! selected. Kernels are compiled from the embedded sources on first use and
//! cached per program key, so repeated dispatches pay the compile cost once per
//! device. Every operation goes through the session's command queue, which is
//! `Send + Sync`, so concurrent dispatch is safe.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::program::Program;
use opencl3::types::cl_device_id;
use sciencekit_common::{SKDeviceInfo, SKError};

use super::availability::unavailable_error;

/// A compiled program, shared by every dispatch that asks for it.
pub(crate) type SharedProgram = Arc<Program>;

/// The device context, command queue and program cache of one backend.
pub(crate) struct SKOpenClSession {
    context: Context,
    queue: CommandQueue,
    device_identifier: usize,
    device: SKDeviceInfo,
    /// The cache key carries the kernel name *and* the scalar type: the same
    /// kernel compiled for `f32` and for `f64` are different programs, so a
    /// single key would hand the second dispatch the first precision's code.
    programs: Mutex<HashMap<String, SharedProgram>>,
}

impl SKOpenClSession {
    /// Open a session on the device at `device_index`.
    pub(crate) fn open(device_index: usize) -> Result<Self, SKError> {
        let identifier = super::device::device_identifiers()?
            .into_iter()
            .find(|(index, _)| *index == device_index)
            .map(|(_, identifier)| identifier)
            .ok_or_else(|| {
                unavailable_error(format!("no OpenCL device with index {device_index}"))
            })?;
        let device = Device::new(identifier);
        let context = Context::from_device(&device).map_err(|code| {
            unavailable_error(format!("OpenCL context creation failed with code {code}"))
        })?;
        let queue = CommandQueue::create_default(&context, 0).map_err(|code| {
            unavailable_error(format!(
                "OpenCL command queue creation failed with code {code}"
            ))
        })?;
        let description = super::device::describe_device(device_index, &device);
        Ok(SKOpenClSession {
            context,
            queue,
            device_identifier: identifier as usize,
            device: description,
            programs: Mutex::new(HashMap::new()),
        })
    }

    /// The selected device description.
    pub(crate) fn device(&self) -> &SKDeviceInfo {
        &self.device
    }

    /// The command queue every transfer and kernel enqueue uses.
    pub(crate) fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// The OpenCL context device storage is allocated in.
    pub(crate) fn context(&self) -> &Context {
        &self.context
    }

    /// The underlying device identifier (for build logs and program builds).
    pub(crate) fn device_identifier(&self) -> cl_device_id {
        // The identifier was stored as a `usize` so the session stays
        // `Send + Sync`: a raw `cl_device_id` (`*mut c_void`) is not, while the
        // handle itself is immutable and valid for the lifetime of the context.
        self.device_identifier as cl_device_id
    }

    /// Compile `source` under `key`, or return the program compiled earlier.
    ///
    /// The caller's key must identify everything that changes the generated
    /// code — the kernel name **and** the scalar type — because the same
    /// kernel compiled for `f32` and `f64` are two different programs.
    ///
    /// A failed build is reported with the driver's build log, never swallowed:
    /// the caller sees the compiler diagnostics and can fall back to the CPU.
    pub(crate) fn program(&self, key: &str, source: &str) -> Result<SharedProgram, SKError> {
        let mut programs = self
            .programs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(program) = programs.get(key) {
            return Ok(Arc::clone(program));
        }
        let program = compile_program(&self.context, self.device_identifier(), source)?;
        let program = Arc::new(program);
        programs.insert(key.to_string(), Arc::clone(&program));
        Ok(program)
    }
}

/// Compile and build one program, turning a build failure into a structured
/// conversion error that carries the driver's build log.
pub(crate) fn compile_program(
    context: &Context,
    device_identifier: cl_device_id,
    source: &str,
) -> Result<Program, SKError> {
    let mut program = Program::create_from_source(context, source).map_err(|code| {
        SKError::Conversion(format!("OpenCL program creation failed with code {code}"))
    })?;
    program.build(&[device_identifier], "").map_err(|code| {
        let build_log = program
            .get_build_log(device_identifier)
            .unwrap_or_else(|_| "the driver reported no build log".to_string());
        SKError::Conversion(format!(
            "OpenCL program build failed with code {code}: {build_log}"
        ))
    })?;
    Ok(program)
}
