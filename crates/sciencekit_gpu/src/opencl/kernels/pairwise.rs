//! Pairwise-distance kernel (spec `opencl-backend`, requirement *OpenCL
//! kernels*).
//!
//! One work item computes one entry of the `left_count × right_count` output:
//! work-item id 0 selects the right-hand row, id 1 the left-hand row. The
//! metric is a kernel argument, so a single compiled program serves all four
//! documented metrics.

use opencl3::kernel::ExecuteKernel;
use opencl3::memory::Buffer;
use opencl3::types::{cl_double, cl_float};
use sciencekit_common::{SKError, SKFloat};

use super::common::create_kernel;
use super::source::PAIRWISE_DISTANCE_SOURCE;
use crate::opencl::buffer::SKOpenClBuffer;
use crate::opencl::session::SKOpenClSession;

/// The distance metric a pairwise computation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKOpenClDistanceMetric {
    /// The sum of squared coordinate differences.
    SquaredEuclidean,
    /// The Euclidean (L2) distance.
    Euclidean,
    /// The Manhattan (L1) distance.
    Manhattan,
    /// The Chebyshev (L∞) distance.
    Chebyshev,
}

impl SKOpenClDistanceMetric {
    /// The documented display name of the metric.
    pub fn name(self) -> &'static str {
        match self {
            SKOpenClDistanceMetric::SquaredEuclidean => "squared-euclidean",
            SKOpenClDistanceMetric::Euclidean => "euclidean",
            SKOpenClDistanceMetric::Manhattan => "manhattan",
            SKOpenClDistanceMetric::Chebyshev => "chebyshev",
        }
    }

    /// The identifier the kernel switches on.
    fn kernel_identifier(self) -> i32 {
        match self {
            SKOpenClDistanceMetric::SquaredEuclidean => 0,
            SKOpenClDistanceMetric::Manhattan => 1,
            SKOpenClDistanceMetric::Chebyshev => 2,
            SKOpenClDistanceMetric::Euclidean => 3,
        }
    }
}

/// Run the pairwise-distance kernel over two resident row-major matrices.
pub(crate) fn run<F: SKFloat>(
    session: &SKOpenClSession,
    left: &SKOpenClBuffer<F>,
    right: &SKOpenClBuffer<F>,
    output: &SKOpenClBuffer<F>,
    metric: SKOpenClDistanceMetric,
) -> Result<(), SKError> {
    let (left_rows, dimensions) = left.shape();
    let (right_rows, right_dimensions) = right.shape();
    if dimensions != right_dimensions {
        return Err(SKError::ShapeMismatch {
            expected: vec![left_rows, dimensions],
            found: vec![right_rows, right_dimensions],
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
            left_rows,
            right_rows,
            dimensions,
            metric,
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
            left_rows,
            right_rows,
            dimensions,
            metric,
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
    left_rows: usize,
    right_rows: usize,
    dimensions: usize,
    metric: SKOpenClDistanceMetric,
    scalar_type: &str,
) -> Result<(), SKError> {
    let kernel = create_kernel(
        session,
        "pairwise-distance",
        PAIRWISE_DISTANCE_SOURCE,
        "sk_pairwise_distance",
        scalar_type,
    )?;
    let mut execution = ExecuteKernel::new(&kernel);
    unsafe {
        execution.set_arg(left);
        execution.set_arg(right);
        execution.set_arg(output);
        execution.set_arg(&(left_rows as u32));
        execution.set_arg(&(right_rows as u32));
        execution.set_arg(&(dimensions as u32));
        execution.set_arg(&metric.kernel_identifier());
    }
    execution.set_global_work_sizes(&[right_rows, left_rows]);
    unsafe { execution.enqueue_nd_range(session.queue()) }.map_err(|code| {
        SKError::UnsupportedOperation {
            operation: "pairwise-distance",
            reason: format!("the OpenCL kernel enqueue failed with code {code}"),
        }
    })?;
    Ok(())
}

fn mixed_precision() -> SKError {
    SKError::Conversion(
        "the pairwise-distance kernel requires all buffers to share one scalar type".to_string(),
    )
}
