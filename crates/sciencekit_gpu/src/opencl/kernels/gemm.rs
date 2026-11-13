//! Dense matrix-multiplication kernel (spec `opencl-backend`, requirement
//! *OpenCL kernels*).
//!
//! The kernel tiles both operands through 16 × 16 local-memory blocks, so each
//! work item reads one left row and one right column from local memory per tile
//! instead of from global memory. Global sizes round up to whole tiles and the
//! kernel bounds-checks every read and the final write.

use opencl3::kernel::ExecuteKernel;
use opencl3::memory::Buffer;
use opencl3::types::{cl_double, cl_float};
use sciencekit_common::{SKError, SKFloat};

use super::common::create_kernel;
use super::source::GEMM_SOURCE;
use crate::opencl::buffer::SKOpenClBuffer;
use crate::opencl::session::SKOpenClSession;

/// The local tile edge the kernel uses; global sizes round up to multiples.
const TILE_EDGE: usize = 16;

/// Run `output = left · right` on the device.
pub(crate) fn run<F: SKFloat>(
    session: &SKOpenClSession,
    left: &SKOpenClBuffer<F>,
    right: &SKOpenClBuffer<F>,
    output: &SKOpenClBuffer<F>,
) -> Result<(), SKError> {
    let (rows, shared_dimension) = left.shape();
    let (right_rows, columns) = right.shape();
    if shared_dimension != right_rows {
        return Err(SKError::ShapeMismatch {
            expected: vec![shared_dimension, columns],
            found: vec![right_rows, columns],
        });
    }
    if output.shape() != (rows, columns) {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows, columns],
            found: vec![output.shape().0, output.shape().1],
        });
    }
    if let (Some(left_buffer), Some(right_buffer), Some(output_buffer)) = (
        left.allocation().single_precision(),
        right.allocation().single_precision(),
        output.allocation().single_precision(),
    ) {
        run_typed::<cl_float>(
            session,
            left_buffer,
            right_buffer,
            output_buffer,
            rows,
            shared_dimension,
            columns,
            "f32",
        )
    } else if let (Some(left_buffer), Some(right_buffer), Some(output_buffer)) = (
        left.allocation().double_precision(),
        right.allocation().double_precision(),
        output.allocation().double_precision(),
    ) {
        run_typed::<cl_double>(
            session,
            left_buffer,
            right_buffer,
            output_buffer,
            rows,
            shared_dimension,
            columns,
            "f64",
        )
    } else {
        Err(mixed_precision())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_typed<T: Copy + Send + Sync + 'static>(
    session: &SKOpenClSession,
    left: &Buffer<T>,
    right: &Buffer<T>,
    output: &Buffer<T>,
    rows: usize,
    shared_dimension: usize,
    columns: usize,
    scalar_type: &str,
) -> Result<(), SKError> {
    let kernel = create_kernel(session, "gemm", GEMM_SOURCE, "sk_gemm_tiled", scalar_type)?;
    let aligned_rows = rows.div_ceil(TILE_EDGE) * TILE_EDGE;
    let aligned_columns = columns.div_ceil(TILE_EDGE) * TILE_EDGE;
    let mut execution = ExecuteKernel::new(&kernel);
    unsafe {
        execution.set_arg(left);
        execution.set_arg(right);
        execution.set_arg(output);
        execution.set_arg(&(rows as u32));
        execution.set_arg(&(shared_dimension as u32));
        execution.set_arg(&(columns as u32));
    }
    execution.set_global_work_sizes(&[aligned_columns, aligned_rows]);
    execution.set_local_work_sizes(&[TILE_EDGE, TILE_EDGE]);
    unsafe { execution.enqueue_nd_range(session.queue()) }.map_err(|code| {
        SKError::UnsupportedOperation {
            operation: "gemm",
            reason: format!("the OpenCL kernel enqueue failed with code {code}"),
        }
    })?;
    Ok(())
}

fn mixed_precision() -> SKError {
    SKError::Conversion("the gemm kernel requires all buffers to share one scalar type".to_string())
}
