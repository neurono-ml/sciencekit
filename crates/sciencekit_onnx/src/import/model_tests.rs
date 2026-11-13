//! Parsing, metadata and typed rejection of malformed models.

use serde_onnx::ir::{
    Dim, ElemType, ExternalData, Graph, Model, Node, OpsetId, Tensor, ValueInfo, ValueType,
};

use super::SKOnnxImportError;
use super::test_support::{affine_regressor_model, identity_model};
use crate::graph::{SKOnnxModel, SKOnnxTensorDType};

#[test]
fn parsed_model_exposes_inputs_outputs_and_counts() {
    let exported = affine_regressor_model();
    let model = SKOnnxModel::from_bytes(exported.bytes().to_vec()).expect("the model parses");
    assert_eq!(model.inputs().len(), 1);
    assert_eq!(model.inputs()[0].name(), "X");
    assert_eq!(model.inputs()[0].element_type(), SKOnnxTensorDType::Double);
    assert_eq!(model.inputs()[0].feature_count(), Some(2));
    assert_eq!(model.outputs().len(), 1);
    assert_eq!(model.outputs()[0].feature_count(), Some(1));
    assert_eq!(model.node_count(), 2);
    assert_eq!(model.initializer_count(), 2);
    assert_eq!(model.opset(), 17);
    assert_eq!(model.onnx_ml_opset(), 3);
}

#[test]
fn invalid_bytes_return_a_structured_error() {
    let error = SKOnnxModel::from_bytes(vec![0xde, 0xad, 0xbe, 0xef]).expect_err("must reject");
    assert!(matches!(error, SKOnnxImportError::InvalidModel { .. }));
}

#[test]
fn an_opset_above_the_supported_ceiling_is_named() {
    let bytes = encode_single_node_model(vec![OpsetId {
        domain: String::new(),
        version: 99,
    }]);
    let error = SKOnnxModel::from_bytes(bytes).expect_err("must reject");
    match error {
        SKOnnxImportError::UnsupportedOpset { declared, .. } => assert_eq!(declared, 99),
        other => panic!("expected a wrong-opset error, got {other}"),
    }
}

#[test]
fn an_external_initializer_is_rejected_by_name() {
    let mut graph = Graph::new("external");
    let matrix = ValueType::tensor(
        ElemType::Double,
        Some(vec![Dim::Param("batch".into()), Dim::Fixed(2)]),
    );
    graph.inputs.push(ValueInfo::new("X", matrix.clone()));
    graph.outputs.push(ValueInfo::new("out", matrix));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["X".into()],
        vec!["out".into()],
        vec![],
    ));
    graph.initializers.push(Tensor::external(
        "weights",
        ElemType::Double,
        vec![Dim::Fixed(1)],
        ExternalData {
            location: "weights.bin".to_string(),
            offset: None,
            length: None,
            checksum: None,
        },
    ));
    let bytes = serde_onnx::proto::encode_model(&Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 17,
        }],
    ));
    let error = SKOnnxModel::from_bytes(bytes).expect_err("must reject");
    assert!(matches!(error, SKOnnxImportError::ExternalData { tensor } if tensor == "weights"));
}

#[test]
fn identity_model_round_trips_through_the_shared_ir() {
    let exported = identity_model();
    let model = SKOnnxModel::from_bytes(exported.bytes().to_vec()).expect("the model parses");
    assert_eq!(model.model().graph.nodes[0].op_type, "Identity");
    assert_eq!(model.model().graph.initializers.len(), 0);
}

fn encode_single_node_model(opsets: Vec<OpsetId>) -> Vec<u8> {
    let mut graph = Graph::new("opset");
    let matrix = ValueType::tensor(
        ElemType::Double,
        Some(vec![Dim::Param("batch".into()), Dim::Fixed(2)]),
    );
    graph.inputs.push(ValueInfo::new("X", matrix.clone()));
    graph.outputs.push(ValueInfo::new("out", matrix));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["X".into()],
        vec!["out".into()],
        vec![],
    ));
    serde_onnx::proto::encode_model(&Model::new(graph, opsets))
}
