//! Elementwise kernels (spec `opencl-backend`, requirement *OpenCL kernels*).
//!
//! Two operations share the same launch shape: an affine transformation
//! (`output = input · factor + offset`) and a two-input combination
//! (`output = left + right`). Both work over contiguous row-major storage, so
//! the matrix shape only decides the element count.

use opencl3::kernel::ExecuteKernel;
use opencl3::memory::Buffer;
use opencl3::types::{cl_double, cl_float};
use sciencekit_common::{SKError, SKFloat};

use super::common::create_kernel;
use super::source::{ELEMENTWISE_COMBINE_SOURCE, ELEMENTWISE_TRANSFORM_SOURCE};
use crate::opencl::buffer::SKOpenClBuffer;
use crate::opencl::session::SKOpenClSession;

/// Run `output = input · factor + offset` elementwise.
pub(crate) fn run_transform<F: SKFloat>(
    session: &SKOpenClSession,
    input: &SKOpenClBuffer<F>,
    output: &SKOpenClBuffer<F>,
    factor: F,
    offset: F,
) -> Result<(), SKError> {
    let element_count = input.element_count_of_shape()?;
    if output.element_count_of_shape()? != element_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![input.shape().0, input.shape().1],
            found: vec![output.shape().0, output.shape().1],
        });
    }
    if let (Some(input_buffer), Some(output_buffer)) = (
        input.allocation().single_precision(),
        output.allocation().single_precision(),
    ) {
        run_transform_typed::<cl_float>(
            session,
            input_buffer,
            output_buffer,
            factor.to_f32().unwrap_or(0.0),
            offset.to_f32().unwrap_or(0.0),
            element_count,
            "f32",
        )
    } else if let (Some(input_buffer), Some(output_buffer)) = (
        input.allocation().double_precision(),
        output.allocation().double_precision(),
    ) {
        run_transform_typed::<cl_double>(
            session,
            input_buffer,
            output_buffer,
            factor.to_f64().unwrap_or(0.0),
            offset.to_f64().unwrap_or(0.0),
            element_count,
            "f64",
        )
    } else {
        Err(mixed_precision())
    }
}

/// Run `output = left + right` elementwise.
pub(crate) fn run_combine<F: SKFloat>(
    session: &SKOpenClSession,
    left: &SKOpenClBuffer<F>,
    right: &SKOpenClBuffer<F>,
    output: &SKOpenClBuffer<F>,
) -> Result<(), SKError> {
    let element_count = left.element_count_of_shape()?;
    if right.element_count_of_shape()? != element_count
        || output.element_count_of_shape()? != element_count
    {
        return Err(SKError::ShapeMismatch {
            expected: vec![left.shape().0, left.shape().1],
            found: vec![right.shape().0, right.shape().1],
        });
    }
    if let (Some(left_buffer), Some(right_buffer), Some(output_buffer)) = (
        left.allocation().single_precision(),
        right.allocation().single_precision(),
        output.allocation().single_precision(),
    ) {
        run_combine_typed::<cl_float>(
            session,
            left_buffer,
            right_buffer,
            output_buffer,
            element_count,
            "f32",
        )
    } else if let (Some(left_buffer), Some(right_buffer), Some(output_buffer)) = (
        left.allocation().double_precision(),
        right.allocation().double_precision(),
        output.allocation().double_precision(),
    ) {
        run_combine_typed::<cl_double>(
            session,
            left_buffer,
            right_buffer,
            output_buffer,
            element_count,
            "f64",
        )
    } else {
        Err(mixed_precision())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_transform_typed<T: Copy + Send + Sync + 'static>(
    session: &SKOpenClSession,
    input: &Buffer<T>,
    output: &Buffer<T>,
    factor: T,
    offset: T,
    element_count: usize,
    scalar_type: &str,
) -> Result<(), SKError> {
    let kernel = create_kernel(
        session,
        "elementwise-transform",
        ELEMENTWISE_TRANSFORM_SOURCE,
        "sk_elementwise_transform",
        scalar_type,
    )?;
    let mut execution = ExecuteKernel::new(&kernel);
    unsafe {
        execution.set_arg(input);
        execution.set_arg(output);
        execution.set_arg(&factor);
        execution.set_arg(&offset);
        execution.set_arg(&(element_count as u32));
    }
    execution.set_global_work_size(element_count);
    unsafe { execution.enqueue_nd_range(session.queue()) }.map_err(|code| {
        SKError::UnsupportedOperation {
            operation: "elementwise-transform",
            reason: format!("the OpenCL kernel enqueue failed with code {code}"),
        }
    })?;
    Ok(())
}

fn run_combine_typed<T: Copy + Send + Sync + 'static>(
    session: &SKOpenClSession,
    left: &Buffer<T>,
    right: &Buffer<T>,
    output: &Buffer<T>,
    element_count: usize,
    scalar_type: &str,
) -> Result<(), SKError> {
    let kernel = create_kernel(
        session,
        "elementwise-combine",
        ELEMENTWISE_COMBINE_SOURCE,
        "sk_elementwise_combine",
        scalar_type,
    )?;
    let mut execution = ExecuteKernel::new(&kernel);
    unsafe {
        execution.set_arg(left);
        execution.set_arg(right);
        execution.set_arg(output);
        execution.set_arg(&(element_count as u32));
    }
    execution.set_global_work_size(element_count);
    unsafe { execution.enqueue_nd_range(session.queue()) }.map_err(|code| {
        SKError::UnsupportedOperation {
            operation: "elementwise-combine",
            reason: format!("the OpenCL kernel enqueue failed with code {code}"),
        }
    })?;
    Ok(())
}

fn mixed_precision() -> SKError {
    SKError::Conversion(
        "the elementwise kernel requires all buffers to share one scalar type".to_string(),
    )
}
