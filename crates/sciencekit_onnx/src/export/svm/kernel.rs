//! The SVM kernel subgraph: `K(X, support_vectors)` as an `[N, S]` tensor.
//!
//! ONNX has no operator for an arbitrary kernel, so each representable kernel is
//! written explicitly. Anything else is rejected by name.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::super::helpers::{float_parameter, integer_parameter, string_parameter, unsupported};

/// Reject a kernel the operator subset cannot express before any tensor is read.
pub(crate) fn require_exportable_kernel<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<(), SKError> {
    match string_parameter(state, "kernel", "rbf") {
        "linear" | "polynomial" | "rbf" | "sigmoid" => Ok(()),
        other => Err(unsupported(
            "SVM kernel",
            format!(
                "the `{other}` kernel cannot be expressed with the exported operator subset; supported kernels are linear, polynomial, rbf and sigmoid"
            ),
        )),
    }
}

/// Append the kernel matrix for the model's configured kernel.
pub(crate) fn append_kernel<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    support_vectors: &str,
    namespace: &str,
) -> Result<String, SKError> {
    let kernel = string_parameter(state, "kernel", "rbf");
    let gamma = float_parameter(state, "gamma", 0.0);
    let degree = integer_parameter(state, "degree", 3).max(1);
    let coef0 = float_parameter(state, "coef0", 0.0);
    match kernel {
        "linear" => append_dot_kernel(builder, input, support_vectors, namespace),
        "sigmoid" => {
            let gram = append_dot_kernel(builder, input, support_vectors, namespace)?;
            let shifted = scale_and_shift::<F>(builder, &gram, gamma, coef0, namespace);
            let output = format!("{namespace}svm.kernel");
            builder.add_node(SKOnnxNode::new("Tanh", vec![shifted], vec![output.clone()]));
            Ok(output)
        }
        "polynomial" => {
            let gram = append_dot_kernel(builder, input, support_vectors, namespace)?;
            let shifted = scale_and_shift::<F>(builder, &gram, gamma, coef0, namespace);
            let exponent = F::from(degree as f64).ok_or(SKError::Conversion(
                "the polynomial degree does not fit the scalar".to_string(),
            ))?;
            builder.add_floating_initializer(
                format!("{namespace}svm.degree"),
                Vec::new(),
                &[exponent],
            );
            let output = format!("{namespace}svm.kernel");
            builder.add_node(SKOnnxNode::new(
                "Pow",
                vec![shifted, format!("{namespace}svm.degree")],
                vec![output.clone()],
            ));
            Ok(output)
        }
        "rbf" => append_rbf_kernel::<F>(builder, input, support_vectors, gamma, namespace),
        other => Err(unsupported(
            "SVM kernel",
            format!(
                "the `{other}` kernel cannot be expressed with the exported operator subset; supported kernels are linear, polynomial, rbf and sigmoid"
            ),
        )),
    }
}

/// `X · SVᵀ`, the shared dot-product kernel.
fn append_dot_kernel(
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    support_vectors: &str,
    namespace: &str,
) -> Result<String, SKError> {
    let transposed = format!("{namespace}svm.support_vectors_transposed");
    builder.add_node(SKOnnxNode::new(
        "Transpose",
        vec![support_vectors.to_string()],
        vec![transposed.clone()],
    ));
    let gram = format!("{namespace}svm.gram");
    builder.add_node(SKOnnxNode::new(
        "MatMul",
        vec![input.to_string(), transposed],
        vec![gram.clone()],
    ));
    Ok(gram)
}

fn scale_and_shift<F: SKFloat>(
    builder: &mut SKOnnxGraphBuilder,
    gram: &str,
    gamma: f64,
    coef0: f64,
    namespace: &str,
) -> String {
    builder.add_floating_initializer(
        format!("{namespace}svm.gamma"),
        Vec::new(),
        &[F::from(gamma).unwrap_or_else(F::zero)],
    );
    builder.add_floating_initializer(
        format!("{namespace}svm.coef0"),
        Vec::new(),
        &[F::from(coef0).unwrap_or_else(F::zero)],
    );
    let scaled = format!("{namespace}svm.scaled");
    builder.add_node(SKOnnxNode::new(
        "Mul",
        vec![gram.to_string(), format!("{namespace}svm.gamma")],
        vec![scaled.clone()],
    ));
    let shifted = format!("{namespace}svm.shifted");
    builder.add_node(SKOnnxNode::new(
        "Add",
        vec![scaled, format!("{namespace}svm.coef0")],
        vec![shifted.clone()],
    ));
    shifted
}

/// `exp(-γ ‖x − s‖²)` through broadcast subtraction and a reduction.
fn append_rbf_kernel<F: SKFloat>(
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    support_vectors: &str,
    gamma: f64,
    namespace: &str,
) -> Result<String, SKError> {
    builder.add_integer_initializer(format!("{namespace}svm.query_axes"), vec![1], &[1]);
    builder.add_integer_initializer(format!("{namespace}svm.reference_axes"), vec![1], &[0]);
    builder.add_integer_initializer(format!("{namespace}svm.distance_axes"), vec![1], &[2]);
    let query = format!("{namespace}svm.query");
    builder.add_node(SKOnnxNode::new(
        "Unsqueeze",
        vec![input.to_string(), format!("{namespace}svm.query_axes")],
        vec![query.clone()],
    ));
    let reference = format!("{namespace}svm.reference");
    builder.add_node(SKOnnxNode::new(
        "Unsqueeze",
        vec![
            support_vectors.to_string(),
            format!("{namespace}svm.reference_axes"),
        ],
        vec![reference.clone()],
    ));
    let difference = format!("{namespace}svm.difference");
    builder.add_node(SKOnnxNode::new(
        "Sub",
        vec![query, reference],
        vec![difference.clone()],
    ));
    let squared = format!("{namespace}svm.squared");
    builder.add_node(SKOnnxNode::new(
        "Mul",
        vec![difference.clone(), difference],
        vec![squared.clone()],
    ));
    let distances = format!("{namespace}svm.distances");
    builder.add_node(
        SKOnnxNode::new(
            "ReduceSum",
            vec![squared, format!("{namespace}svm.distance_axes")],
            vec![distances.clone()],
        )
        .attribute_integer("keepdims", 0),
    );
    let negated = format!("{namespace}svm.negated");
    builder.add_node(SKOnnxNode::new(
        "Neg",
        vec![distances],
        vec![negated.clone()],
    ));
    builder.add_floating_initializer(
        format!("{namespace}svm.gamma"),
        Vec::new(),
        &[F::from(gamma).ok_or(SKError::Conversion("gamma is not finite".to_string()))?],
    );
    let scaled = format!("{namespace}svm.scaled");
    builder.add_node(SKOnnxNode::new(
        "Mul",
        vec![negated, format!("{namespace}svm.gamma")],
        vec![scaled.clone()],
    ));
    let output = format!("{namespace}svm.kernel");
    builder.add_node(SKOnnxNode::new("Exp", vec![scaled], vec![output.clone()]));
    Ok(output)
}
