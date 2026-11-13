//! Malformed shape metadata must produce a typed error, never a panic.
//!
//! These fixtures are decodable ONNX graphs whose declared shapes disagree with
//! learned tensors or whose axis lists are invalid. A foreign model can carry
//! any of them, so the interpreter must validate before calling an ndarray
//! routine that panics on the mismatch.

use ndarray::array;
use sciencekit_common::SKRegressorPredictor;

use super::SKOnnxImportError;
use super::SKOnnxPredictor;
use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};

#[test]
fn an_initializer_width_that_differs_from_the_declared_input_width_is_rejected() {
    let mut builder = SKOnnxGraphBuilder::new("initializer_width");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_floating_initializer("coefficients", vec![3, 1], &[1.0_f64, 2.0, 3.0])
        .add_node(SKOnnxNode::new(
            "MatMul",
            vec!["X".to_string(), "coefficients".to_string()],
            vec!["out".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Double, 1)
                .expect("a valid output"),
        );
    let model = builder.build().expect("the malformed graph still encodes");
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(model)
        .build()
        .expect("the interface is analysable");
    let error = predictor
        .predict(array![[1.0, 2.0]].view())
        .expect_err("the inner dimensions disagree");
    match error {
        SKOnnxImportError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(found, vec![3, 1]);
        }
        other => panic!("expected a shape-mismatch error, got {other}"),
    }
}

#[test]
fn a_transpose_permutation_of_the_wrong_length_is_rejected() {
    let error = run_operator(
        SKOnnxNode::new("Transpose", vec!["X".to_string()], vec!["out".to_string()])
            .attribute_integers("perm", vec![0]),
    );
    assert!(matches!(error, SKOnnxImportError::InvalidOperator { .. }));
}

#[test]
fn a_transpose_permutation_with_repeated_axes_is_rejected() {
    let error = run_operator(
        SKOnnxNode::new("Transpose", vec!["X".to_string()], vec!["out".to_string()])
            .attribute_integers("perm", vec![0, 0]),
    );
    assert!(matches!(error, SKOnnxImportError::InvalidOperator { .. }));
}

#[test]
fn squeezing_the_same_axis_twice_is_rejected() {
    let error = run_operator(
        SKOnnxNode::new("Squeeze", vec!["X".to_string()], vec!["out".to_string()])
            .attribute_integers("axes", vec![1, 1]),
    );
    assert!(matches!(error, SKOnnxImportError::InvalidOperator { .. }));
}

#[test]
fn a_reshape_whose_dimensions_overflow_is_rejected() {
    let error = run_operator(
        SKOnnxNode::new("Reshape", vec!["X".to_string()], vec!["out".to_string()])
            .attribute_integers("shape", vec![i64::MAX, 4]),
    );
    assert!(matches!(
        error,
        SKOnnxImportError::InvalidOperator { .. } | SKOnnxImportError::ShapeMismatch { .. }
    ));
}

/// Run one node between a two-column input and a two-column output.
fn run_operator(node: SKOnnxNode) -> SKOnnxImportError {
    let mut builder = SKOnnxGraphBuilder::new("malformed_shape");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(node)
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Double, 2)
                .expect("a valid output"),
        );
    let model = builder.build().expect("the malformed graph still encodes");
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(model)
        .build()
        .expect("the interface is analysable");
    predictor
        .predict(array![[1.0, 2.0]].view())
        .expect_err("the malformed shape must be rejected")
}
