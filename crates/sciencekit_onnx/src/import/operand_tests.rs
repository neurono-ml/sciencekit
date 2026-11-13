//! Malformed nodes must fail with a typed error, never panic.
//!
//! A foreign model can declare a supported operator with fewer inputs than the
//! operator requires. The dispatcher must validate the operand position before
//! indexing instead of trusting the protobuf.

use ndarray::array;
use sciencekit_common::SKRegressorPredictor;

use super::SKOnnxImportError;
use super::SKOnnxPredictor;
use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo};

#[test]
fn matmul_without_its_right_operand_is_rejected_instead_of_panicking() {
    let mut builder = SKOnnxGraphBuilder::new("arity");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "MatMul",
            vec!["X".to_string()],
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
        .expect_err("the missing operand must be rejected");
    match error {
        SKOnnxImportError::InvalidOperator { operator, detail } => {
            assert_eq!(operator, "MatMul");
            assert!(detail.contains("position 1"), "{detail}");
        }
        other => panic!("expected an invalid-operator error, got {other}"),
    }
}

#[test]
fn an_empty_operand_slot_is_rejected_instead_of_being_looked_up() {
    let mut builder = SKOnnxGraphBuilder::new("empty_operand");
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "Add",
            vec!["X".to_string(), String::new()],
            vec!["out".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::floating_matrix("out", SKOnnxTensorDType::Double, 2)
                .expect("a valid output"),
        );
    let model = builder.build().expect("the malformed graph still encodes");
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(model)
        .build()
        .expect("the interface is analysable");
    let error = predictor
        .predict(array![[1.0, 2.0]].view())
        .expect_err("the empty operand must be rejected");
    assert!(matches!(error, SKOnnxImportError::InvalidOperator { .. }));
}
