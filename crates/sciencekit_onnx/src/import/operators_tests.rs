//! The documented operator contract and its typed rejection.

use super::SKOnnxImportError;
use super::{SK_ONNX_SUPPORTED_OPERATORS, sk_onnx_supports_operator};
use crate::graph::{
    SKOnnxDimension, SKOnnxGraphBuilder, SKOnnxModel, SKOnnxNode, SKOnnxTensorDType,
    SKOnnxValueInfo,
};

#[test]
fn every_documented_operator_is_advertised_once() {
    let mut sorted = SK_ONNX_SUPPORTED_OPERATORS.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), SK_ONNX_SUPPORTED_OPERATORS.len());
    for operator in ["MatMul", "ReduceSum", "TopK", "Softmax", "Cast", "Gather"] {
        assert!(
            sk_onnx_supports_operator("", operator),
            "{operator} must be supported"
        );
    }
    assert!(sk_onnx_supports_operator(
        "ai.onnx.ml",
        "TreeEnsembleClassifier"
    ));
    assert!(!sk_onnx_supports_operator("", "Cos"));
}

#[test]
fn an_unsupported_operator_is_rejected_with_its_name() {
    let model = unsupported_graph("Cos", "");
    let error = SKOnnxModel::from_bytes(model.bytes().to_vec()).expect_err("must reject");
    match error {
        SKOnnxImportError::UnsupportedOperator {
            operator, domain, ..
        } => {
            assert_eq!(operator, "Cos");
            assert_eq!(domain, "");
        }
        other => panic!("expected an unsupported-operator error, got {other}"),
    }
}

#[test]
fn an_unsupported_ml_operator_is_rejected_with_its_domain() {
    let model = unsupported_graph("Scaler", "ai.onnx.ml");
    let error = SKOnnxModel::from_bytes(model.bytes().to_vec()).expect_err("must reject");
    match error {
        SKOnnxImportError::UnsupportedOperator {
            operator, domain, ..
        } => {
            assert_eq!(operator, "Scaler");
            assert_eq!(domain, "ai.onnx.ml");
        }
        other => panic!("expected an unsupported-operator error, got {other}"),
    }
}

fn unsupported_graph(operator: &str, domain: &str) -> SKOnnxModel {
    let mut builder = SKOnnxGraphBuilder::new("unsupported");
    let node = if domain.is_empty() {
        SKOnnxNode::new(operator, vec!["X".to_string()], vec!["out".to_string()])
    } else {
        SKOnnxNode::ml(operator, vec!["X".to_string()], vec!["out".to_string()])
    };
    builder
        .add_input(
            SKOnnxValueInfo::floating_matrix("X", SKOnnxTensorDType::Double, 2)
                .expect("a valid input"),
        )
        .add_node(node)
        .add_output(
            SKOnnxValueInfo::new(
                "out",
                SKOnnxTensorDType::Double,
                vec![SKOnnxDimension::batch(), SKOnnxDimension::Fixed(1)],
            )
            .expect("a valid output"),
        );
    builder.build().expect("the fixture graph is valid")
}
