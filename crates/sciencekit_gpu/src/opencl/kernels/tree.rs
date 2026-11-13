//! Tree-ensemble prediction kernel (spec `opencl-backend`, requirement *OpenCL
//! kernels*).
//!
//! One work item walks one feature row from the root to a leaf. The tree is a
//! dense node matrix with six columns per node: feature index, threshold, left
//! child index, right child index, leaf value and the leaf flag (`1` for a
//! leaf, `0` for a split). A negative flag never occurs; only an explicit `1`
//! marks a leaf.

use opencl3::kernel::ExecuteKernel;
use opencl3::memory::Buffer;
use opencl3::types::{cl_double, cl_float};
use sciencekit_common::{SKError, SKFloat};

use super::common::create_kernel;
use super::source::TREE_PREDICT_SOURCE;
use crate::opencl::buffer::SKOpenClBuffer;
use crate::opencl::session::SKOpenClSession;

/// The number of columns one tree node occupies in the node matrix.
pub const SK_OPENCL_TREE_NODE_COLUMNS: usize = 6;

/// Run the tree-prediction kernel over `features` and a node matrix.
pub(crate) fn run<F: SKFloat>(
    session: &SKOpenClSession,
    features: &SKOpenClBuffer<F>,
    nodes: &SKOpenClBuffer<F>,
    output: &SKOpenClBuffer<F>,
) -> Result<(), SKError> {
    let (rows, feature_count) = features.shape();
    let (node_count, node_columns) = nodes.shape();
    if node_columns != SK_OPENCL_TREE_NODE_COLUMNS {
        return Err(SKError::ShapeMismatch {
            expected: vec![node_count, SK_OPENCL_TREE_NODE_COLUMNS],
            found: vec![node_count, node_columns],
        });
    }
    if output.shape() != (rows, 1) {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows, 1],
            found: vec![output.shape().0, output.shape().1],
        });
    }
    if let (Some(feature_buffer), Some(node_buffer), Some(output_buffer)) = (
        features.allocation().single_precision(),
        nodes.allocation().single_precision(),
        output.allocation().single_precision(),
    ) {
        run_typed::<cl_float>(
            session,
            feature_buffer,
            node_buffer,
            output_buffer,
            rows,
            feature_count,
            node_count,
            "f32",
        )
    } else if let (Some(feature_buffer), Some(node_buffer), Some(output_buffer)) = (
        features.allocation().double_precision(),
        nodes.allocation().double_precision(),
        output.allocation().double_precision(),
    ) {
        run_typed::<cl_double>(
            session,
            feature_buffer,
            node_buffer,
            output_buffer,
            rows,
            feature_count,
            node_count,
            "f64",
        )
    } else {
        Err(mixed_precision())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_typed<T: Copy + Send + Sync + 'static>(
    session: &SKOpenClSession,
    features: &Buffer<T>,
    nodes: &Buffer<T>,
    output: &Buffer<T>,
    rows: usize,
    feature_count: usize,
    node_count: usize,
    scalar_type: &str,
) -> Result<(), SKError> {
    let kernel = create_kernel(
        session,
        "tree-predict",
        TREE_PREDICT_SOURCE,
        "sk_tree_predict",
        scalar_type,
    )?;
    let mut execution = ExecuteKernel::new(&kernel);
    unsafe {
        execution.set_arg(features);
        execution.set_arg(nodes);
        execution.set_arg(output);
        execution.set_arg(&(rows as u32));
        execution.set_arg(&(feature_count as u32));
        execution.set_arg(&(node_count as u32));
    }
    execution.set_global_work_size(rows);
    unsafe { execution.enqueue_nd_range(session.queue()) }.map_err(|code| {
        SKError::UnsupportedOperation {
            operation: "tree-predict",
            reason: format!("the OpenCL kernel enqueue failed with code {code}"),
        }
    })?;
    Ok(())
}

fn mixed_precision() -> SKError {
    SKError::Conversion(
        "the tree-predict kernel requires all buffers to share one scalar type".to_string(),
    )
}
