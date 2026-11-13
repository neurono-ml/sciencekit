//! Missing producers, ambiguous heads and dtype validation.

use ndarray::array;
use sciencekit_common::SKRegressorPredictor;

use super::SKOnnxImportError;
use super::SKOnnxPredictor;
use serde_onnx::ir::{Dim, ElemType, Node, Tensor, TensorData, ValueInfo, ValueType};

use crate::graph::{
    SKOnnxGraphBuilder, SKOnnxModel, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo,
};

#[test]
fn a_node_reading_an_undefined_value_names_the_missing_tensor() {
    let bytes = encode_graph(
        &[ValueInfo::new("X", matrix())],
        &[ValueInfo::new("out", matrix())],
        vec![Node::new(
            "Add",
            "",
            vec!["X".to_string(), "ghost".to_string()],
            vec!["out".to_string()],
            vec![],
        )],
        vec![],
    );
    let model = SKOnnxModel::from_bytes(bytes).expect("the interface is decodable");
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model(model)
        .build()
        .expect("the interface is still decodable");
    let error = predictor
        .predict(array![[1.0, 2.0]].view())
        .expect_err("must fail");
    match error {
        SKOnnxImportError::MissingInitializer { name, .. } => assert_eq!(name, "ghost"),
        other => panic!("expected a missing-initializer error, got {other}"),
    }
}

#[test]
fn an_integer_feature_input_is_rejected_as_an_unsupported_dtype() {
    let mut builder = SKOnnxGraphBuilder::new("integer_input");
    builder
        .add_input(
            SKOnnxValueInfo::new(
                "X",
                SKOnnxTensorDType::Int64,
                vec![
                    crate::graph::SKOnnxDimension::batch(),
                    crate::graph::SKOnnxDimension::Fixed(2),
                ],
            )
            .expect("a valid input"),
        )
        .add_node(SKOnnxNode::new(
            "Identity",
            vec!["X".to_string()],
            vec!["out".to_string()],
        ))
        .add_output(
            SKOnnxValueInfo::new(
                "out",
                SKOnnxTensorDType::Int64,
                vec![crate::graph::SKOnnxDimension::batch()],
            )
            .expect("a valid output"),
        );
    let model = builder.build().expect("the fixture graph is valid");
    let error = SKOnnxModel::from_bytes(model.bytes().to_vec()).expect_err("must reject");
    assert!(matches!(error, SKOnnxImportError::UnsupportedDtype { .. }));
}

#[test]
fn a_graph_without_a_required_input_is_rejected() {
    let bytes = encode_graph(
        &[],
        &[ValueInfo::new("out", matrix())],
        vec![Node::new(
            "Identity",
            "",
            vec!["weights".to_string()],
            vec!["out".to_string()],
            vec![],
        )],
        vec![Tensor::new(
            "weights",
            ElemType::Double,
            vec![Dim::Fixed(1), Dim::Fixed(2)],
            TensorData::F64(vec![1.0, 2.0]),
        )],
    );
    let error = SKOnnxModel::from_bytes(bytes).expect_err("must reject");
    assert!(matches!(error, SKOnnxImportError::UnsupportedOutput { .. }));
}

fn matrix() -> ValueType {
    ValueType::tensor(
        ElemType::Double,
        Some(vec![Dim::Param("batch".into()), Dim::Fixed(2)]),
    )
}

fn encode_graph(
    inputs: &[ValueInfo],
    outputs: &[ValueInfo],
    nodes: Vec<Node>,
    initializers: Vec<Tensor>,
) -> Vec<u8> {
    let mut graph = serde_onnx::ir::Graph::new("validation");
    graph.inputs = inputs.to_vec();
    graph.outputs = outputs.to_vec();
    graph.nodes = nodes;
    graph.initializers = initializers;
    serde_onnx::proto::encode_model(&serde_onnx::ir::Model::new(
        graph,
        vec![serde_onnx::ir::OpsetId {
            domain: String::new(),
            version: 17,
        }],
    ))
}
