//! The support-vector dual expansion: `K(x, s) · α + b`.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::super::helpers::floating_tensor;
use super::kernel::append_kernel;

/// Append `MatMul(kernel, dual_coefficients) + bias` and return the margin name.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_expansion<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    support_vectors: &str,
    dual_coefficients: &[F],
    bias: F,
    namespace: &str,
) -> Result<String, SKError> {
    let kernel = append_kernel(state, builder, input, support_vectors, namespace)?;
    builder.add_floating_initializer(
        format!("{namespace}svm.dual_coefficients"),
        vec![dual_coefficients.len(), 1],
        dual_coefficients,
    );
    let expansion = format!("{namespace}svm.expansion");
    builder.add_node(SKOnnxNode::new(
        "MatMul",
        vec![kernel, format!("{namespace}svm.dual_coefficients")],
        vec![expansion.clone()],
    ));
    builder.add_floating_initializer(format!("{namespace}svm.bias"), vec![1, 1], &[bias]);
    let output = format!("{namespace}svm.margin");
    builder.add_node(SKOnnxNode::new(
        "Add",
        vec![expansion, format!("{namespace}svm.bias")],
        vec![output.clone()],
    ));
    Ok(output)
}

/// Read a `[rows, features]` support-vector block and one scalar from a state.
pub(crate) fn read_dense_support_vectors<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
) -> Result<(Vec<usize>, Vec<F>), SKError> {
    floating_tensor(state, &format!("{prefix}.support_vectors_dense"))
}

/// Read the single-element bias tensor.
pub(crate) fn read_bias<F: SKFloat>(state: &SKModelState<F>, prefix: &str) -> Result<F, SKError> {
    let (_, bias_values) = floating_tensor(state, &format!("{prefix}.bias"))?;
    let [bias] = bias_values.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![1],
            found: vec![bias_values.len()],
        });
    };
    Ok(*bias)
}
